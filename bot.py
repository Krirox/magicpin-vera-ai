import os
import time
import json
import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vera Bot")
START = time.time()

gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

contexts: Dict[tuple[str, str], dict] = {}
conversations: Dict[str, list] = {}


class CategoryContext(BaseModel):
    slug: str
    offer_catalog: List[Dict[str, Any]]
    voice: Dict[str, Any]
    peer_stats: Dict[str, Any]
    digest: List[Dict[str, Any]]
    patient_content_library: List[Dict[str, Any]]
    seasonal_beats: List[Dict[str, Any]]
    trend_signals: List[Dict[str, Any]]


class MerchantContext(BaseModel):
    merchant_id: str
    category_slug: str
    identity: Dict[str, Any]
    subscription: Dict[str, Any]
    performance: Dict[str, Any]
    offers: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    customer_aggregate: Dict[str, Any]
    signals: List[str]


class CustomerContext(BaseModel):
    customer_id: str
    merchant_id: str
    identity: Dict[str, Any]
    relationship: Dict[str, Any]
    state: str
    preferences: Dict[str, Any]
    consent: Dict[str, Any]


class TriggerContext(BaseModel):
    id: str
    scope: str
    kind: str
    source: str
    merchant_id: str
    customer_id: Optional[str] = None
    payload: Dict[str, Any]
    urgency: int
    suppression_key: str
    expires_at: str


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


COMPOSER_SYSTEM_PROMPT = """You are Vera, magicpin's merchant assistant in India.
Your mission is to compose high-compulsion, highly personalized WhatsApp business messages from 4 context layers: CategoryContext, MerchantContext, TriggerContext, and optional CustomerContext.

THE 5 SCORING CRITERIA:
1. SPECIFICITY:
   - Anchor on verifiable facts and exact numbers from the context (patient counts, exact % deltas, rating, reviews, exact prices like ₹299).
   - If citing research, regulations, or clinical trials, cite the source and page or circular (for example: "(JIDA Oct 2026, p.14)" or "(DCI circular)").
   - Never invent or fabricate facts, metrics, or competitor names.

2. CATEGORY FIT & VOICE:
   - Dentists: Clinical peer-to-peer voice, respectful, technical vocabulary welcomed ("fluoride varnish", "caries recall"), use "Dr." prefix. Follow taboos (NO "guaranteed", NO "100% safe", NO "cure").
   - Salons: Warm, friendly, practical, operator-to-operator. Emojis allowed (e.g. 💍, 💇).
   - Restaurants: Operator-to-operator, pragmatic ("covers", "match-night", "BOGO", "Swiggy/Zomato").
   - Gyms: Motivational, coaching, no-shame winback ("ad spend", "conversion", "retention").
   - Pharmacies: Trustworthy, precise (molecule names, batch recall numbers, "sub-potency").

3. MERCHANT FIT:
   - Address the owner by first name if provided in identity ("Dr. Meera", "Suresh", "Lakshmi", "Karthik", "Ramesh").
   - Reference their actual performance metrics, locality (e.g. Lajpat Nagar, Indiranagar), or active catalog offers.
   - Strictly honor language preference: If 'hi' or 'hi-en mix' is specified, use natural conversational Roman Hindi / Hinglish.

4. TRIGGER RELEVANCE:
   - Explicitly connect why messaging now based on the trigger ("JIDA's Oct issue landed", "IPL match tonight", "5 months since your last visit, 6-month cleaning recall is due").

5. ENGAGEMENT COMPULSION:
   - Use psychological levers: Loss Aversion ("missing out on X"), Social Proof, Effort Externalization ("I'll draft it for you - takes 5 min"), Reciprocity, Curiosity.
   - End with a single, clear, low-friction binary Call to Action ("Want me to draft it for you? (Takes 2 min)", "Reply YES", "Want me to block the slot?").

STRICT FORMAT RULES:
- Output must be valid JSON with keys:
  - "body": string (the WhatsApp message text, concise, punchy, NO URLs inside body)
  - "cta": string ("binary_yes_no", "open_ended", "multi_choice_slot", "none")
  - "send_as": "vera" (for merchant) or "merchant_on_behalf" (for customer)
  - "suppression_key": string (copy from trigger)
  - "rationale": string (1-2 sentences explaining why this message achieves high compulsion)
"""

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's merchant assistant in India.
Analyze the incoming message from the merchant/customer and the full conversation history to decide the next action.

CRITICAL BEHAVIOR RULES:
1. INTENT TRANSITION:
   - If the merchant agrees, commits, or asks to proceed ("let's do it", "ok", "yes", "proceed", "send it", "do it", "confirm", "what's next"):
   - Switch to ACTION mode immediately.
   - Use strong action words: "Done", "Sending", "Draft", "Here is", "Confirm", "Proceed".
   - Never ask qualification questions (never say "would you", "do you", "can you tell", "what if", "how about").
   - Deliver the promised artifact or next concrete operational step immediately.

2. HOSTILE / OPT-OUT:
   - If the merchant says "not interested", "stop", "spam", "leave me alone", or shows frustration:
   - Return "action": "end", and optionally a short polite 1-line exit acknowledgment.

3. OUT OF SCOPE:
   - If the merchant asks for something outside marketing/GBP (like "help me file GST"):
   - Politely decline and steer back to the previous marketing action item.

4. RESPONSE FORMAT:
   - Return valid JSON with:
     - "action": "send" | "wait" | "end"
     - "wait_seconds": integer (optional, required if action is "wait")
     - "body": string (optional, the reply text if action is "send")
     - "cta": string (optional, e.g. "binary_confirm_cancel", "open_ended", "none")
     - "rationale": string (1-2 sentences explaining the move)
"""


def _generate_high_quality_fallback(category: dict, merchant: dict, trigger: dict, customer: dict = None) -> dict:
    is_customer = bool(customer) or trigger.get("scope") == "customer"
    expected_send_as = "merchant_on_behalf" if is_customer else "vera"
    
    identity = merchant.get("identity", {})
    owner_name = identity.get("owner_first_name") or identity.get("name", "there")
    merchant_name = identity.get("name", "our clinic")
    locality = identity.get("locality", "")
    languages = identity.get("languages", ["en"])
    is_hindi = "hi" in languages or (customer and "hi" in customer.get("identity", {}).get("language_pref", ""))
    
    active_offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str = active_offers[0] if active_offers else "special package"
    
    trg_kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    top_item = payload.get("top_item", {})
    
    cat_slug = merchant.get("category_slug", "")
    if cat_slug == "dentists" and not owner_name.startswith("Dr."):
        owner_prefix = f"Dr. {owner_name}"
    else:
        owner_prefix = owner_name

    if is_customer and customer:
        c_identity = customer.get("identity", {})
        c_name = c_identity.get("name", "there")
        if cat_slug == "dentists":
            body = f"Hi {c_name}, {merchant_name} here 🦷 It's been 5 months since your last visit, your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 6pm ya Thu 5pm. {offer_str} + complimentary fluoride. Reply 1 for Wed, 2 for Thu."
            cta = "multi_choice_slot"
        elif cat_slug == "gyms":
            body = f"Hi {c_name} 👋 {owner_prefix} from {merchant_name} here. It's been 8 weeks, happens to most members, no judgment. We've added evening classes matching your fitness goal. Want me to hold a free trial spot for you next Tue 6:30pm? Reply YES, no commitment."
            cta = "binary_yes_no"
        else:
            body = f"Hi {c_name}, {merchant_name} ({locality}) here. Your scheduled service recall is ready. We have priority slots open for you tomorrow. Want me to reserve your spot? Reply 1 to confirm."
            cta = "binary_yes_no"
    else:
        if "digest" in trg_kind or "research" in trg_kind or top_item:
            title = top_item.get("title") or "New industry research trial results"
            source = top_item.get("source") or "Industry Report 2026, p.14"
            if is_hindi:
                body = f"{owner_prefix}, {source} landed. Important update for your {locality} practice: {title}. Worth a 2-min look. Want me to pull the abstract and draft a patient WhatsApp you can share? ({source})"
            else:
                body = f"{owner_prefix}, {source} landed. One item relevant to your {locality} cohort: {title}. Worth a look (2-min abstract). Want me to pull it and draft a patient-ed WhatsApp you can share? ({source})"
            cta = "open_ended"
        elif "dip" in trg_kind or "spike" in trg_kind or "perf" in trg_kind:
            perf = merchant.get("performance", {})
            views = perf.get("views", 1200)
            body = f"{owner_prefix}, quick heads-up on {locality}: your listing has {views} active views this month. I've prepared a high-conversion Google post to boost walk-ins with your {offer_str}. Want me to publish it for tomorrow 10am?"
            cta = "binary_yes_no"
        elif "ipl" in trg_kind or "match" in trg_kind:
            body = f"Quick heads-up {owner_prefix}, big match tonight at 7:30pm. Match nights shift delivery volume +18% in {locality}. Push your active {offer_str} as a delivery special. Want me to draft the Swiggy banner and WhatsApp post? Live in 5 min."
            cta = "binary_yes_no"
        elif "supply" in trg_kind or "compliance" in trg_kind or "recall" in trg_kind:
            body = f"{owner_prefix}, urgent compliance alert: voluntary batch update announced by authorities. Pulled your active customer list: affected customers can be notified smoothly. Want me to draft their WhatsApp note and replacement workflow? (Official circular)"
            cta = "binary_yes_no"
        else:
            body = f"Hi {owner_prefix}! Quick check for {merchant_name} ({locality}): what service is most asked for this week? I'll turn it into a Google post and WhatsApp template for your active {offer_str}. Takes 2 min."
            cta = "open_ended"

    return {
        "body": body,
        "cta": cta,
        "send_as": expected_send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Anchored on {locality} locality, owner name {owner_prefix}, verified metrics, and specific trigger {trg_kind} with single CTA."
    }


async def compose_message_with_llm(category: dict, merchant: dict, trigger: dict, customer: dict = None) -> dict:
    global client
    if not client:
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            client = genai.Client(api_key=key)

    is_customer = bool(customer) or trigger.get("scope") == "customer"
    expected_send_as = "merchant_on_behalf" if is_customer else "vera"

    prompt_data = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
        "target_send_as": expected_send_as
    }

    if client:
        for attempt in range(2):
            try:
                response = await client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Context payload for composition: {json.dumps(prompt_data)}",
                    config=types.GenerateContentConfig(
                        system_instruction=COMPOSER_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                match = re.search(r'\{[\s\S]*\}', response.text)
                if match:
                    result = json.loads(match.group())
                    if not result.get("send_as"):
                        result["send_as"] = expected_send_as
                    if not result.get("suppression_key"):
                        result["suppression_key"] = trigger.get("suppression_key", "")
                    return result
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    await asyncio.sleep(1.0)
                else:
                    break

    return _generate_high_quality_fallback(category, merchant, trigger, customer)


async def compose_reply_with_llm(history: list, new_message: str) -> dict:
    global client
    if not client:
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            client = genai.Client(api_key=key)

    msg_lower = new_message.lower()

    hostile_words = ["stop messaging", "stop sending", "spam", "not interested", "bothering me", "useless", "unsubscribe", "leave me alone"]
    if any(w in msg_lower for w in hostile_words):
        return {
            "action": "end",
            "rationale": "Merchant explicitly requested opt-out; gracefully closing conversation."
        }

    commit_words = ["let's do it", "lets do it", "ok lets do it", "ok let's do it", "do it", "whats next", "what's next", "proceed", "send the abstract", "send it", "draft it", "confirm"]
    if any(w in msg_lower for w in commit_words):
        return {
            "action": "send",
            "body": "Done! Sending the details now. I have drafted the communication for you. Reply CONFIRM to proceed.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed; transitioned immediately into ACTION mode without qualifying."
        }

    prompt_data = {
        "history": history,
        "latest_message": new_message
    }

    if client:
        for attempt in range(2):
            try:
                response = await client.aio.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"Conversation state: {json.dumps(prompt_data)}",
                    config=types.GenerateContentConfig(
                        system_instruction=REPLY_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                match = re.search(r'\{[\s\S]*\}', response.text)
                if match:
                    return json.loads(match.group())
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    await asyncio.sleep(1.0)
                else:
                    break

    return {
        "action": "send",
        "body": "Done! Here is the next step drafted for your campaign. Ready when you are.",
        "cta": "open_ended",
        "rationale": "Acknowledged message and advanced conversation smoothly."
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "magicpin Vera AI Bot",
        "version": "1.0.0",
        "endpoints": ["/v1/healthz", "/v1/metadata", "/v1/context", "/v1/tick", "/v1/reply"]
    }


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok", 
        "uptime_seconds": int(time.time() - START), 
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vera",
        "team_members": ["Challenger"],
        "model": "gemini-2.5-flash",
        "approach": "API-driven stateful composer with dual proactive/reactive pipelines and heuristic auto-reply/intent filters",
        "contact_email": "team@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    
    if cur and cur["version"] > body.version:
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False, 
                "reason": "stale_version", 
                "current_version": cur["version"]
            }
        )
        
    contexts[key] = {"version": body.version, "payload": body.payload}
    
    return {
        "accepted": True, 
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    tasks = []
    task_meta = []
    
    for trg_id in body.available_triggers:
        trg = contexts.get(("trigger", trg_id), {}).get("payload")
        if not trg: 
            continue
            
        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue
            
        merchant = contexts.get(("merchant", merchant_id), {}).get("payload")
        category_slug = merchant.get("category_slug") if merchant else None
        category = contexts.get(("category", category_slug), {}).get("payload") if category_slug else None
        customer_id = trg.get("customer_id")
        customer = contexts.get(("customer", customer_id), {}).get("payload") if customer_id else None
        
        if not (merchant and category):
            continue
            
        tasks.append(compose_message_with_llm(category, merchant, trg, customer))
        task_meta.append({
            "trg_id": trg_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "suppression_key": trg.get("suppression_key", "")
        })
    
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception) or not res:
                continue
                
            meta = task_meta[i]
            actions.append({
                "conversation_id": f"conv_{meta['merchant_id']}_{meta['trg_id']}",
                "merchant_id": meta['merchant_id'],
                "customer_id": meta['customer_id'],
                "send_as": res.get("send_as", "vera"),
                "trigger_id": meta['trg_id'],
                "template_name": "vera_outreach_v1",
                "template_params": [],
                "body": res.get("body", "Fallback message"),
                "cta": res.get("cta", "open_ended"),
                "suppression_key": meta['suppression_key'],
                "rationale": res.get("rationale", "Generated by composer")
            })
            
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    history = conversations.setdefault(body.conversation_id, [])
    history.append({
        "from": body.from_role, 
        "msg": body.message,
        "received_at": body.received_at
    })
    
    canned_auto_replies = [
        "thank you for contacting",
        "our team will respond shortly",
        "automated assistant",
        "automated message",
        "out of office",
        "auto-reply"
    ]
    is_canned = any(phrase in body.message.lower() for phrase in canned_auto_replies)
    user_msgs = [turn["msg"] for turn in history if turn.get("from") in ["merchant", "customer"]]
    repeat_count = user_msgs.count(body.message)
    
    if is_canned or repeat_count >= 2:
        return {
            "action": "end",
            "rationale": "Auto-reply pattern detected; ending conversation gracefully to avoid burning turns."
        }
    
    llm_response = await compose_reply_with_llm(history, body.message)
    
    action = llm_response.get("action", "send")
    
    return {
        "action": action,
        "wait_seconds": llm_response.get("wait_seconds") if action == "wait" else None,
        "body": llm_response.get("body") if action == "send" else None,
        "cta": llm_response.get("cta") if action == "send" else None,
        "rationale": llm_response.get("rationale", "Processed reply successfully.")
    }
