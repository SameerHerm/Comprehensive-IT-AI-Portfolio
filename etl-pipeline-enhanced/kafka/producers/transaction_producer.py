"""
Transaction Data Producer for Kafka
Generates and sends transaction data to Kafka topics
"""

import json
import time
import random
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from kafka import KafkaProducer
from faker import Faker
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionProducer:
    """Producer for transaction data"""
    
    def __init__(self, bootstrap_servers: str = None):
        """Initialize transaction producer"""
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.faker = Faker()
        
        # Transaction categories and products
        self.categories = {
            'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Smart Watch', 'Camera'],
            'Clothing': ['Shirt', 'Pants', 'Dress', 'Shoes', 'Jacket', 'Accessories'],
            'Groceries': ['Fruits', 'Vegetables', 'Dairy', 'Meat', 'Beverages', 'Snacks'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Magazines'],
            'Home': ['Furniture', 'Decor', 'Kitchen', 'Bedding', 'Storage', 'Lighting']
        }
        
        self.payment_methods = ['credit_card', 'debit_card', 'paypal', 'cash', 'bank_transfer', 'crypto']
        self.transaction_status = ['completed', 'pending', 'failed', 'cancelled', 'refunded']
        
    def generate_transaction(self) -> Dict[str, Any]:
        """Generate a single transaction"""
        category = random.choice(list(self.categories.keys()))
        product = random.choice(self.categories[category])
        
        # Generate price based on category
        price_ranges = {
            'Electronics': (50, 3000),
            'Clothing': (10, 500),
            'Groceries': (1, 100),
            'Books': (5, 50),
            'Home': (20, 2000)
        }
        
        min_price, max_price = price_ranges[category]
        price = round(random.uniform(min_price, max_price), 2)
        quantity = random.randint(1, 5)
        
        # Generate customer data
        customer_id = f"CUST_{random.randint(10000, 99999)}"
        
        # Generate transaction
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_id,
            'customer_name': self.faker.name(),
            'customer_email': self.faker.email(),
            'customer_phone': self.faker.phone_number(),
            'customer_address': {
                'street': self.faker.street_address(),
                'city': self.faker.city(),
                'state': self.faker.state(),
                'zip_code': self.faker.zipcode(),
                'country': self.faker.country()
            },
            'product_category': category,
            'product_name': product,
            'product_id': f"PROD_{category[:3].upper()}_{random.randint(1000, 9999)}",
            'quantity': quantity,
            'unit_price': price,
            'total_amount': round(price * quantity, 2),
            'discount': round(random.uniform(0, 0.3) * price * quantity, 2) if random.random() > 0.7 else 0,
            'tax': round(price * quantity * 0.08, 2),  # 8% tax
            'payment_method': random.choice(self.payment_methods),
            'transaction_status': random.choice(self.transaction_status),
            'currency': 'USD',
            'store_id': f"STORE_{random.randint(1, 50):03d}",
            'cashier_id': f"EMP_{random.randint(100, 999)}",
            'pos_terminal': f"POS_{random.randint(1, 20):02d}",
            'loyalty_points': random.randint(0, 100) if random.random() > 0.5 else 0,
            'is_online': random.choice([True, False]),
            'device_type': random.choice(['web', 'mobile', 'pos']) if random.choice([True, False]) else 'pos',
            'session_id': str(uuid.uuid4()) if random.choice([True, False]) else None
        }
        
        # Calculate final amount
        transaction['final_amount'] = round(
            transaction['total_amount'] - transaction['discount'] + transaction['tax'], 2
        )
        
        # Add fraud detection fields
        transaction['fraud_score'] = round(random.uniform(0, 1), 3)
        transaction['is_suspicious'] = transaction['fraud_score'] > 0.8
        
        return transaction
    
    def generate_batch(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Generate a batch of transactions"""
        return [self.generate_transaction() for _ in range(batch_size)]
    
    def produce_transactions(self, topic: str = 'transactions', 
                           batch_size: int = 10, 
                           interval: int = 1,
                           max_transactions: int = None):
        """Continuously produce transaction data"""
        logger.info(f"Starting transaction production to topic: {topic}")
        
        transaction_count = 0
        
        try:
            while max_transactions is None or transaction_count < max_transactions:
                batch = self.generate_batch(batch_size)
                
                for transaction in batch:
                    key = transaction['customer_id']
                    
                    future = self.producer.send(topic, value=transaction, key=key)
                    
                    try:
                        record_metadata = future.get(timeout=10)
                        transaction_count += 1
                        
                        if transaction_count % 100 == 0:
                            logger.info(f"Produced {transaction_count} transactions")
                            
                    except Exception as e:
                        logger.error(f"Failed to send transaction: {e}")
                
                self.producer.flush()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Transaction producer stopped by user")
        finally:
            self.close()
        
        logger.info(f"Total transactions produced: {transaction_count}")
    
    def produce_fraud_transactions(self, topic: str = 'fraud-transactions'):
        """Generate and produce potentially fraudulent transactions"""
        logger.info("Generating fraudulent transaction patterns...")
        
        # Generate suspicious patterns
        fraud_patterns = [
            self.generate_high_value_transaction,
            self.generate_rapid_transactions,
            self.generate_unusual_location_transaction,
            self.generate_multiple_payment_methods
        ]
        
        for pattern_func in fraud_patterns:
            transactions = pattern_func()
            for transaction in transactions:
                self.producer.send(topic, value=transaction)
        
        self.producer.flush()
        logger.info("Fraudulent transactions sent")
    
    def generate_high_value_transaction(self) -> List[Dict[str, Any]]:
        """Generate unusually high value transactions"""
        transactions = []
        for _ in range(5):
            transaction = self.generate_transaction()
            transaction['total_amount'] = round(random.uniform(10000, 50000), 2)
            transaction['fraud_score'] = round(random.uniform(0.8, 1.0), 3)
            transaction['is_suspicious'] = True
            transaction['fraud_type'] = 'high_value'
            transactions.append(transaction)
        return transactions
    
    def generate_rapid_transactions(self) -> List[Dict[str, Any]]:
        """Generate rapid successive transactions"""
        customer_id = f"CUST_{random.randint(10000, 99999)}"
        transactions = []
        
        base_time = datetime.now()
        for i in range(10):
            transaction = self.generate_transaction()
            transaction['customer_id'] = customer_id
            transaction['timestamp'] = (base_time + timedelta(seconds=i*30)).isoformat()
            transaction['fraud_score'] = round(random.uniform(0.7, 0.95), 3)
            transaction['is_suspicious'] = True
            transaction['fraud_type'] = 'rapid_succession'
            transactions.append(transaction)
        
        return transactions
    
    def generate_unusual_location_transaction(self) -> List[Dict[str, Any]]:
        """Generate transactions from unusual locations"""
        transaction = self.generate_transaction()
        transaction['customer_address']['country'] = random.choice(['Nigeria', 'Romania', 'Russia'])
        transaction['fraud_score'] = round(random.uniform(0.75, 0.95), 3)
        transaction['is_suspicious'] = True
        transaction['fraud_type'] = 'unusual_location'
        return [transaction]
    
    def generate_multiple_payment_methods(self) -> List[Dict[str, Any]]:
        """Generate transactions with multiple payment methods"""
        customer_id = f"CUST_{random.randint(10000, 99999)}"
        transactions = []
        
        for payment_method in self.payment_methods[:4]:
            transaction = self.generate_transaction()
            transaction['customer_id'] = customer_id
            transaction['payment_method'] = payment_method
            transaction['fraud_score'] = round(random.uniform(0.6, 0.85), 3)
            transaction['is_suspicious'] = True
            transaction['fraud_type'] = 'multiple_payment_methods'
            transactions.append(transaction)
        
        return transactions
    
    def close(self):
        """Close producer connection"""
        self.producer.close()
        logger.info("Transaction producer closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transaction Data Producer')
    parser.add_argument('--topic', default='transactions', help='Kafka topic')
    parser.add_argument('--servers', default='localhost:9092', help='Bootstrap servers')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--interval', type=int, default=1, help='Interval between batches')
    parser.add_argument('--max-transactions', type=int, help='Maximum transactions to produce')
    parser.add_argument('--fraud', action='store_true', help='Generate fraud transactions')
    
    args = parser.parse_args()
    
    producer = TransactionProducer(bootstrap_servers=args.servers)
    
    if args.fraud:
        producer.produce_fraud_transactions()
    else:
        producer.produce_transactions(
            topic=args.topic,
            batch_size=args.batch_size,
            interval=args.interval,
            max_transactions=args.max_transactions
        )

if __name__ == "__main__":
    main()
