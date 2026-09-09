"""
Data loading module for Twitter customer support dataset.
Loads from Kaggle and handles multi-turn conversation structure.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


class DataLoader:
    """Load and preprocess customer support conversations from Kaggle Twitter dataset."""
    
    def __init__(self, brand: str = "amazon", raw_data_path: str = "data/raw/"):
        self.brand = brand.lower()
        self.raw_data_path = Path(raw_data_path)
        self.logger = logging.getLogger(__name__)
        
    def load_sample(self, sample_size: int = 1500) -> List[Dict[str, Any]]:
        """
        Load stratified sample of conversations from the Twitter dataset.
        
        Args:
            sample_size: Number of conversation turns to sample
            
        Returns:
            List of conversation dictionaries with message, intent labels, history, etc.
        """
        
        # Try to load from CSV first (simulated dataset for reproducibility)
        csv_path = self.raw_data_path / f"{self.brand}_conversations.csv"
        
        if not csv_path.exists():
            self.logger.warning(f"No local dataset found at {csv_path}")
            self.logger.info("Using simulated data for demonstration")
            return self._generate_mock_data(sample_size)
        
        # Load real data from CSV
        self.logger.info(f"Loading conversations for {self.brand} from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Sample stratified by intent if available
        if "intent" in df.columns:
            sample = df.groupby("intent", group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max(1, sample_size // df["intent"].nunique())),
                                   random_state=42)
            ).sample(n=min(len(df), sample_size), random_state=42)
        else:
            sample = df.sample(n=min(len(df), sample_size), random_state=42)
        
        # Convert to conversation format
        conversations = []
        for idx, row in sample.iterrows():
            conv = {
                "id": row.get("conversation_id", f"conv_{idx}"),
                "message": row.get("text", row.get("message", "")),
                "author": row.get("author_id", row.get("author", "")),
                "created_at": row.get("created_at", ""),
                "true_intent": row.get("intent", None),
                "history": self._extract_history(row),
                "brand": self.brand
            }
            conversations.append(conv)
        
        self.logger.info(f"Loaded {len(conversations)} conversations")
        return conversations
    
    def _extract_history(self, row: pd.Series) -> Dict[str, Any]:
        """Extract customer history from row (order info, past interactions, etc.)."""
        
        return {
            "order_count": row.get("order_count", 0),
            "account_age_days": row.get("account_age_days", 0),
            "previous_complaints": row.get("previous_complaints", 0),
            "vip_status": row.get("vip_status", False),
            "order_value": row.get("order_value", 0),
            "previous_unresolved_tickets": row.get("unresolved_tickets", 0)
        }
    
    def _generate_mock_data(self, sample_size: int) -> List[Dict[str, Any]]:
        """Generate mock data for testing/demo when real dataset not available."""
        
        self.logger.info(f"Generating {sample_size} mock conversations for {self.brand}")
        
        intents = [
            "order_status", "shipping_delay", "refund_request", "return_request",
            "billing_issue", "payment_failed", "product_quality", "technical_issue",
            "account_problem", "complaint", "lost_package", "discount_inquiry",
            "general_inquiry", "warranty_claim", "login_issue"
        ]
        
        mock_messages = {
            "order_status": [
                "Where's my order?",
                "Can you track my package?",
                "When will my delivery arrive?",
                "Has my order shipped yet?",
                "What's the status of order #12345?"
            ],
            "shipping_delay": [
                "My order hasn't arrived after 2 weeks",
                "Why is my delivery so late?",
                "Your shipping is incredibly slow",
                "I ordered this 3 weeks ago and still no delivery",
                "My package is overdue"
            ],
            "refund_request": [
                "I want a refund",
                "Can I get my money back?",
                "Please refund my order",
                "How do I get a refund?",
                "I need to be refunded for this purchase"
            ],
            "complaint": [
                "Your customer service is terrible",
                "I'm very disappointed with this experience",
                "This is unacceptable",
                "Worst purchase ever",
                "I'm so angry about this"
            ],
            "product_quality": [
                "This product is broken",
                "The quality is terrible",
                "It arrived damaged",
                "This doesn't work at all",
                "Very poor quality"
            ],
            "billing_issue": [
                "I was charged twice",
                "Why is this on my bill?",
                "I don't recognize this charge",
                "Incorrect billing amount",
                "Duplicate charge detected"
            ],
            "general_inquiry": [
                "Do you ship internationally?",
                "What are your hours?",
                "How do I contact support?",
                "What's your return policy?",
                "Do you have this in stock?"
            ]
        }
        
        conversations = []
        for i in range(sample_size):
            intent = random.choice(intents)
            message = random.choice(mock_messages.get(intent, ["General question"]))
            
            conv = {
                "id": f"conv_{i}",
                "message": message,
                "author": f"user_{i}",
                "created_at": "2024-01-15",
                "true_intent": intent,
                "history": {
                    "order_count": random.randint(0, 20),
                    "account_age_days": random.randint(1, 1000),
                    "previous_complaints": random.randint(0, 5),
                    "vip_status": random.random() > 0.9,
                    "order_value": random.randint(0, 1000),
                    "previous_unresolved_tickets": random.randint(0, 3)
                },
                "brand": self.brand
            }
            conversations.append(conv)
        
        self.logger.info(f"Generated {len(conversations)} mock conversations")
        return conversations
    
    def load_golden_dataset(self, path: str = "data/golden_dataset.csv") -> pd.DataFrame:
        """Load the hand-labelled golden evaluation dataset."""
        
        csv_path = Path(path)
        if not csv_path.exists():
            self.logger.error(f"Golden dataset not found at {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        self.logger.info(f"Loaded golden dataset with {len(df)} examples")
        return df
