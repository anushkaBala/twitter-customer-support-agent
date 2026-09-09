"""
Reply generation module that grounds responses in historical resolutions.
"""

import json
import logging
from typing import Dict, List, Any, Optional
import os

import openai


class ReplyGenerator:
    """Generate contextual replies grounded in brand's historical patterns."""
    
    def __init__(self, brand: str = "amazon", model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7, brand_config: Dict = None):
        self.brand = brand
        self.model = model
        self.temperature = temperature
        self.brand_config = brand_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Setup OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Load brand voice
        self.brand_voice = self.brand_config.get("voice", {})
        self.policies = self.brand_config.get("policies", {})
        self.guardrails = self.brand_config.get("guardrails", [])
    
    def generate(self, message: str, intent: str, customer_history: Dict = None) -> Dict[str, Any]:
        """
        Generate a contextual reply to customer message.
        
        Args:
            message: Customer message
            intent: Classified intent
            customer_history: Customer's order/account history
            
        Returns:
            {
                "reply": "Thank you for contacting us...",
                "grounding": {
                    "similar_case_id": "conv_12847",
                    "resolution_pattern": "..."
                },
                "quality_score": 0.85
            }
        """
        
        customer_history = customer_history or {}
        
        # Build context-aware prompt
        prompt = self._build_generation_prompt(message, intent, customer_history)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a customer support agent for {self.brand}. "
                                 f"Tone: {self.brand_voice.get('tone', 'helpful')}. "
                                 f"Keep replies concise (1-3 sentences)."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            
            reply_text = response.choices[0].message.content.strip()
            
            # Apply guardrails
            reply_text = self._apply_guardrails(reply_text, intent)
            
            return {
                "reply": reply_text,
                "grounding": self._get_grounding(intent),
                "quality_score": self._estimate_quality(reply_text, intent)
            }
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {
                "reply": f"Thank you for contacting {self.brand}. Our team will assist you shortly.",
                "grounding": {},
                "quality_score": 0.5
            }
    
    def _build_generation_prompt(self, message: str, intent: str, 
                                 customer_history: Dict) -> str:
        """Build prompt with context for reply generation."""
        
        # Get resolution pattern for this intent
        patterns = self.brand_config.get("resolution_patterns", {})
        intent_pattern = patterns.get(intent, ["Provide helpful, empathetic response"])
        
        pattern_text = "\n".join([f"- {p}" for p in intent_pattern[:2]])
        
        prompt = f"""Customer Message: "{message}"

Intent Category: {intent}

Customer Background:
- Customer since: {customer_history.get('account_age_days', 0)} days
- Previous complaints: {customer_history.get('previous_complaints', 0)}
- VIP Status: {customer_history.get('vip_status', False)}

Resolution Approach for {intent}:
{pattern_text}

Generate a helpful, empathetic reply that:
1. Acknowledges the customer's issue
2. Provides specific next steps
3. Maintains a {self.brand_voice.get('tone', 'professional')} tone

Reply:"""
        
        return prompt
    
    def _apply_guardrails(self, reply: str, intent: str) -> str:
        """Apply safety guardrails to generated reply."""
        
        # Check guardrails
        if "guarantee" in reply.lower() and intent == "shipping_delay":
            # Don't guarantee delivery dates
            reply = reply.replace("guarantee", "estimate")
            reply = reply.replace("will definitely", "should")
        
        # Limit financial commitments
        if "$" in reply and intent not in ["refund_request", "billing_issue"]:
            # Don't promise money without approval
            lines = reply.split(".")
            filtered = [l for l in lines if "$" not in l]
            reply = ".".join(filtered) if filtered else reply
        
        # Limit length
        if len(reply) > 500:
            reply = reply[:497] + "..."
        
        return reply.strip()
    
    def _get_grounding(self, intent: str) -> Dict[str, Any]:
        """Get grounding info (similar historical case, pattern, etc.)."""
        
        # In production, would retrieve from vector DB
        # For now, return static patterns
        patterns = self.brand_config.get("resolution_patterns", {})
        
        return {
            "intent": intent,
            "resolution_pattern": patterns.get(intent, ["Standard response"])[0],
            "similar_case_id": f"case_{intent[:4]}_0001",
            "confidence": 0.8
        }
    
    def _estimate_quality(self, reply: str, intent: str) -> float:
        """Estimate quality of generated reply."""
        
        score = 0.7  # Base score
        
        # Bonus for including specific details
        if any(kw in reply.lower() for kw in ["order", "tracking", "refund"]):
            score += 0.1
        
        # Bonus for length (not too short, not too long)
        if 20 < len(reply) < 300:
            score += 0.1
        
        # Penalty for generic responses
        if any(generic in reply.lower() for generic in ["please wait", "thank you for your patience"]):
            score -= 0.05
        
        return min(1.0, max(0.5, score))
    
    def batch_generate(self, messages: List[str], intents: List[str]) -> List[Dict]:
        """Generate replies for multiple messages."""
        
        results = []
        for msg, intent in zip(messages, intents):
            results.append(self.generate(msg, intent))
        return results
