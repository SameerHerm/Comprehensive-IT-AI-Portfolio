"""
Weather Data Producer for Kafka
Generates and sends weather data to Kafka topics
"""

import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from kafka import KafkaProducer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherProducer:
    """Producer for weather data"""
    
    def __init__(self, bootstrap_servers: str = None):
        """Initialize weather producer"""
        self.bootstrap_servers = bootstrap_servers or os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.api_key = os.getenv('WEATHER_API_KEY', '')
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        # List of cities to get weather data for
        self.cities = [
            {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
            {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437},
            {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298},
            {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698},
            {'name': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740}
        ]
    
    def fetch_real_weather(self, city: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch real weather data from API"""
        if not self.api_key:
            return self.generate_mock_weather(city)
        
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': city['lat'],
                'lon': city['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'timestamp': datetime.now().isoformat(),
                    'city': city['name'],
                    'temperature': data['main']['temp'],
                    'feels_like': data['main']['feels_like'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'wind_direction': data['wind'].get('deg', 0),
                    'clouds': data['clouds']['all'],
                    'weather': data['weather'][0]['main'],
                    'description': data['weather'][0]['description']
                }
            else:
                logger.warning(f"API request failed: {response.status_code}")
                return self.generate_mock_weather(city)
                
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self.generate_mock_weather(city)
    
    def generate_mock_weather(self, city: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock weather data"""
        weather_conditions = ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm', 'Drizzle', 'Mist']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'city': city['name'],
            'temperature': round(random.uniform(-10, 40), 1),
            'feels_like': round(random.uniform(-15, 45), 1),
            'humidity': random.randint(20, 95),
            'pressure': random.randint(980, 1040),
            'wind_speed': round(random.uniform(0, 25), 1),
            'wind_direction': random.randint(0, 360),
            'clouds': random.randint(0, 100),
            'weather': random.choice(weather_conditions),
            'description': f"{random.choice(['light', 'moderate', 'heavy'])} {random.choice(weather_conditions).lower()}",
            'uv_index': round(random.uniform(0, 11), 1),
            'visibility': random.randint(1000, 10000),
            'precipitation': round(random.uniform(0, 50), 1) if random.random() > 0.7 else 0
        }
    
    def produce_weather_data(self, topic: str = 'weather-data', interval: int = 60):
        """Continuously produce weather data"""
        logger.info(f"Starting weather data production to topic: {topic}")
        
        try:
            while True:
                for city in self.cities:
                    weather_data = self.fetch_real_weather(city)
                    
                    # Add additional calculated fields
                    weather_data['heat_index'] = self.calculate_heat_index(
                        weather_data['temperature'],
                        weather_data['humidity']
                    )
                    weather_data['wind_chill'] = self.calculate_wind_chill(
                        weather_data['temperature'],
                        weather_data['wind_speed']
                    )
                    
                    # Send to Kafka
                    key = city['name'].lower().replace(' ', '_')
                    future = self.producer.send(topic, value=weather_data, key=key)
                    
                    try:
                        record_metadata = future.get(timeout=10)
                        logger.info(f"Weather data for {city['name']} sent to {record_metadata.topic}")
                    except Exception as e:
                        logger.error(f"Failed to send weather data: {e}")
                
                # Flush and wait
                self.producer.flush()
                logger.info(f"Weather batch sent. Waiting {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Weather producer stopped by user")
        finally:
            self.close()
    
    def calculate_heat_index(self, temp: float, humidity: float) -> float:
        """Calculate heat index"""
        if temp < 27:  # Heat index is relevant for temps >= 80°F (27°C)
            return temp
        
        # Simplified heat index formula
        hi = temp + 0.5 * (humidity + temp - 30)
        return round(hi, 1)
    
    def calculate_wind_chill(self, temp: float, wind_speed: float) -> float:
        """Calculate wind chill"""
        if temp > 10 or wind_speed < 4.8:  # Wind chill relevant for cold temps
            return temp
        
        # Simplified wind chill formula
        wc = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
        return round(wc, 1)
    
    def close(self):
        """Close producer connection"""
        self.producer.close()
        logger.info("Weather producer closed")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weather Data Producer')
    parser.add_argument('--topic', default='weather-data', help='Kafka topic')
    parser.add_argument('--servers', default='localhost:9092', help='Bootstrap servers')
    parser.add_argument('--interval', type=int, default=60, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    producer = WeatherProducer(bootstrap_servers=args.servers)
    producer.produce_weather_data(topic=args.topic, interval=args.interval)

if __name__ == "__main__":
    main()
