"""
Enhanced Healthcare & Wellness Locator Agent with JSON Response Format - FIXED

Key fixes:
1. Fixed AttributeError by ensuring _generate_response_text receives Facility objects
2. Limited results to 10 locations as requested
3. Improved error handling and data flow
"""

import os
import json
import logging
import asyncio
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, List, Optional, Any, Tuple, Union
from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import math

# Configure logging
logger = logging.getLogger(__name__)

class FacilityCategory(Enum):
    """Comprehensive facility categories"""
    # Healthcare Facilities
    HOSPITAL = "hospital"
    EMERGENCY_ROOM = "emergency_room"
    URGENT_CARE = "urgent_care"
    MEDICAL_CLINIC = "medical_clinic"
    SPECIALTY_CLINIC = "specialty_clinic"
    
    # Rehabilitation & Therapy
    PHYSICAL_THERAPY = "physical_therapy"
    OCCUPATIONAL_THERAPY = "occupational_therapy"
    SPEECH_THERAPY = "speech_therapy"
    REHABILITATION_CENTER = "rehabilitation_center"
    
    # Neurological & Brain Health
    NEUROLOGY_CLINIC = "neurology_clinic"
    NEUROTHERAPY_CLINIC = "neurotherapy_clinic"
    BRAIN_INJURY_CENTER = "brain_injury_center"
    COGNITIVE_THERAPY = "cognitive_therapy"
    
    # Mental Health & Wellness
    MENTAL_HEALTH_CLINIC = "mental_health_clinic"
    PSYCHOLOGY_PRACTICE = "psychology_practice"
    PSYCHIATRY_CLINIC = "psychiatry_clinic"
    COUNSELING_CENTER = "counseling_center"
    SUPPORT_GROUP_CENTER = "support_group_center"
    
    # Fitness & Exercise
    GYM = "gym"
    FITNESS_CENTER = "fitness_center"
    YOGA_STUDIO = "yoga_studio"
    PILATES_STUDIO = "pilates_studio"
    PERSONAL_TRAINING = "personal_training"
    CROSSFIT_GYM = "crossfit_gym"
    MARTIAL_ARTS = "martial_arts"
    DANCE_STUDIO = "dance_studio"
    
    # Wellness & Holistic Health
    WELLNESS_CENTER = "wellness_center"
    SPA = "spa"
    MASSAGE_THERAPY = "massage_therapy"
    ACUPUNCTURE = "acupuncture"
    CHIROPRACTIC = "chiropractic"
    MEDITATION_CENTER = "meditation_center"
    HOLISTIC_HEALTH = "holistic_health"
    NATUROPATHY = "naturopathy"
    
    # Specialized Wellness
    NUTRITION_COUNSELING = "nutrition_counseling"
    WEIGHT_LOSS_CENTER = "weight_loss_center"
    ADDICTION_RECOVERY = "addiction_recovery"
    SENIOR_FITNESS = "senior_fitness"
    ADAPTIVE_FITNESS = "adaptive_fitness"

class UrgencyLevel(Enum):
    """Service urgency levels"""
    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"
    WELLNESS = "wellness"

class LocationType(Enum):
    """Location specification types"""
    ZIP_CODE = "zip_code"
    CITY = "city"
    ADDRESS = "address"
    CURRENT_LOCATION = "current_location"
    COORDINATES = "coordinates"

@dataclass
class SearchCriteria:
    """Comprehensive search criteria from LLM parsing"""
    original_query: str
    facility_categories: List[FacilityCategory]
    location: str
    location_type: LocationType
    urgency: UrgencyLevel
    radius_miles: int
    specific_services: List[str] = field(default_factory=list)
    accessibility_needs: List[str] = field(default_factory=list)
    insurance_accepted: List[str] = field(default_factory=list)
    price_preference: Optional[str] = None
    availability_preference: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""

@dataclass
class Facility:
    """Comprehensive facility information"""
    # Basic Information
    name: str
    category: str
    description: str = ""
    
    # Location Details
    address: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    distance_miles: Optional[float] = None
    
    # Contact Information
    phone: Optional[str] = None
    website: Optional[str] = None
    email: Optional[str] = None
    
    # Quality Indicators
    rating: Optional[float] = None
    review_count: Optional[int] = None
    
    # Business Details
    business_status: str = "OPERATIONAL"
    hours: Optional[Dict[str, str]] = None
    price_level: Optional[int] = None
    
    # Services & Features
    services: List[str] = field(default_factory=list)
    accessibility_features: List[str] = field(default_factory=list)
    insurance_accepted: List[str] = field(default_factory=list)
    
    # External Links
    maps_url: str = ""
    booking_url: Optional[str] = None
    
    # Metadata
    place_id: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert facility to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO string
        if isinstance(data.get('last_updated'), datetime):
            data['last_updated'] = data['last_updated'].isoformat()
        return data

class SearchParser(ABC):
    """Abstract base class for search query parsers"""
    
    @abstractmethod
    async def parse_query(self, query: str) -> SearchCriteria:
        """Parse natural language query into structured search criteria"""
        pass

class LLMSearchParser(SearchParser):
    """Advanced LLM-powered search query parser"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def parse_query(self, query: str) -> SearchCriteria:
        """Parse natural language query using advanced LLM intelligence"""
        
        parsing_prompt = self._build_parsing_prompt(query)
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    parsing_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            result_text = self._clean_json_response(response.text)
            result = json.loads(result_text)
            
            return self._convert_to_search_criteria(query, result)
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            # Return basic search criteria instead of failing
            return self._create_basic_criteria(query)
    
    def _build_parsing_prompt(self, query: str) -> str:
        """Build comprehensive parsing prompt for LLM"""
        
        return f"""
You are an expert facility locator query parser. Analyze this user query and extract detailed search criteria for finding healthcare, wellness, and fitness facilities.

USER QUERY: "{query}"

TASK: Extract comprehensive search information including facility types, location, preferences, and special requirements.

FACILITY CATEGORIES (choose all relevant):
**Healthcare:**
- "hospital" - General hospitals, medical centers
- "emergency_room" - Emergency departments, urgent medical care
- "urgent_care" - Walk-in clinics, immediate care
- "medical_clinic" - Primary care, family practice
- "specialty_clinic" - Cardiology, oncology, dermatology, etc.

**Rehabilitation & Therapy:**
- "physical_therapy" - PT clinics, sports medicine
- "occupational_therapy" - OT centers, workplace injury recovery
- "speech_therapy" - Speech pathology, communication disorders
- "rehabilitation_center" - Comprehensive rehab facilities

**Neurological & Brain Health:**
- "neurology_clinic" - Neurologists, brain specialists
- "neurotherapy_clinic" - Brain injury treatment
- "brain_injury_center" - TBI specialized centers
- "cognitive_therapy" - Memory, cognitive rehabilitation

**Mental Health & Wellness:**
- "mental_health_clinic" - General mental health services
- "psychology_practice" - Psychologists, therapy
- "psychiatry_clinic" - Psychiatrists, medication management
- "counseling_center" - Individual and group counseling
- "support_group_center" - Peer support, group meetings

**Fitness & Exercise:**
- "gym" - Traditional gyms, weight training
- "fitness_center" - Full-service fitness facilities
- "yoga_studio" - Yoga classes, meditation
- "pilates_studio" - Pilates classes, core fitness
- "personal_training" - One-on-one fitness coaching
- "crossfit_gym" - CrossFit boxes, high-intensity training
- "martial_arts" - Karate, jiu-jitsu, boxing
- "dance_studio" - Dance classes, movement therapy

**Wellness & Holistic Health:**
- "wellness_center" - Comprehensive wellness services
- "spa" - Relaxation, beauty treatments
- "massage_therapy" - Therapeutic massage, bodywork
- "acupuncture" - Traditional Chinese medicine
- "chiropractic" - Spinal health, alignment
- "meditation_center" - Mindfulness, spiritual wellness
- "holistic_health" - Alternative medicine, natural healing
- "naturopathy" - Natural medicine practitioners

**Specialized Wellness:**
- "nutrition_counseling" - Dietitians, nutrition coaching
- "weight_loss_center" - Weight management programs
- "addiction_recovery" - Substance abuse treatment
- "senior_fitness" - Fitness for older adults
- "adaptive_fitness" - Disability-friendly fitness

LOCATION TYPES:
- "zip_code" - 5-digit ZIP code
- "city" - City name with/without state
- "address" - Street address
- "current_location" - Near user's current location

URGENCY LEVELS:
- "emergency" - Immediate medical attention needed
- "urgent" - Same day or next day service needed
- "routine" - Standard appointment scheduling
- "wellness" - General wellness, no time pressure

DEFAULT VALUES:
- If no location specified: "current_location"
- If no urgency specified: "wellness" for fitness/wellness, "routine" for healthcare
- Default radius: 15 miles for wellness/fitness, 25 miles for healthcare

Respond in JSON format:
{{
    "facility_categories": ["list", "of", "relevant", "categories"],
    "location": "extracted location string",
    "location_type": "zip_code/city/address/current_location",
    "urgency": "emergency/urgent/routine/wellness",
    "radius_miles": number,
    "specific_services": ["specific", "services", "mentioned"],
    "accessibility_needs": ["accessibility", "requirements"],
    "insurance_accepted": ["insurance", "types", "if", "mentioned"],
    "price_preference": "free/low-cost/any/null",
    "availability_preference": "24-7/weekends/evenings/flexible/null",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of parsing decisions"
}}

JSON Response:"""
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean and extract JSON from LLM response"""
        text = response_text.strip()
        
        # Remove code block markers
        if text.startswith('```json'):
            text = text[7:-3]
        elif text.startswith('```'):
            text = text[3:-3]
        
        return text.strip()
    
    def _convert_to_search_criteria(self, query: str, result: Dict) -> SearchCriteria:
        """Convert LLM result to SearchCriteria object"""
        
        # Convert category strings to enums
        categories = []
        for category_str in result.get("facility_categories", []):
            try:
                categories.append(FacilityCategory(category_str))
            except ValueError:
                logger.warning(f"Unknown facility category: {category_str}")
        
        if not categories:
            categories = [FacilityCategory.MEDICAL_CLINIC]
        
        # Convert location type
        location_type_str = result.get("location_type", "current_location")
        try:
            location_type = LocationType(location_type_str)
        except ValueError:
            location_type = LocationType.CURRENT_LOCATION
        
        # Convert urgency level
        urgency_str = result.get("urgency", "routine")
        try:
            urgency = UrgencyLevel(urgency_str)
        except ValueError:
            urgency = UrgencyLevel.ROUTINE
        
        return SearchCriteria(
            original_query=query,
            facility_categories=categories,
            location=result.get("location", ""),
            location_type=location_type,
            urgency=urgency,
            radius_miles=result.get("radius_miles", 15),
            specific_services=result.get("specific_services", []),
            accessibility_needs=result.get("accessibility_needs", []),
            insurance_accepted=result.get("insurance_accepted", []),
            price_preference=result.get("price_preference"),
            availability_preference=result.get("availability_preference"),
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", "")
        )
    
    def _create_basic_criteria(self, query: str) -> SearchCriteria:
        """Create basic search criteria when LLM parsing fails"""
        return SearchCriteria(
            original_query=query,
            facility_categories=[FacilityCategory.MEDICAL_CLINIC],
            location="",
            location_type=LocationType.CURRENT_LOCATION,
            urgency=UrgencyLevel.ROUTINE,
            radius_miles=15,
            confidence=0.1,
            reasoning="Basic fallback due to parsing error"
        )

class PlacesAPIConnector:
    """Enhanced Google Places API connector with comprehensive facility support"""
    
    def __init__(self, api_key: str):
        self.api_key = os.getenv("GOOGLE_API_KEY", api_key)
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.session = requests.Session()
        
        # Comprehensive Google Places API type mappings
        self.api_type_mappings = {
            # Healthcare
            FacilityCategory.HOSPITAL: ["hospital"],
            FacilityCategory.EMERGENCY_ROOM: ["hospital"],
            FacilityCategory.URGENT_CARE: ["doctor", "health"],
            FacilityCategory.MEDICAL_CLINIC: ["doctor", "health"],
            FacilityCategory.SPECIALTY_CLINIC: ["doctor", "health"],
            
            # Therapy & Rehabilitation
            FacilityCategory.PHYSICAL_THERAPY: ["physiotherapist", "health"],
            FacilityCategory.OCCUPATIONAL_THERAPY: ["physiotherapist", "health"],
            FacilityCategory.SPEECH_THERAPY: ["health"],
            FacilityCategory.REHABILITATION_CENTER: ["physiotherapist", "health"],
            
            # Neurological
            FacilityCategory.NEUROLOGY_CLINIC: ["doctor", "health"],
            FacilityCategory.NEUROTHERAPY_CLINIC: ["doctor", "health"],
            FacilityCategory.BRAIN_INJURY_CENTER: ["health"],
            FacilityCategory.COGNITIVE_THERAPY: ["health"],
            
            # Mental Health
            FacilityCategory.MENTAL_HEALTH_CLINIC: ["health"],
            FacilityCategory.PSYCHOLOGY_PRACTICE: ["health"],
            FacilityCategory.PSYCHIATRY_CLINIC: ["doctor", "health"],
            FacilityCategory.COUNSELING_CENTER: ["health"],
            FacilityCategory.SUPPORT_GROUP_CENTER: ["health"],
            
            # Fitness
            FacilityCategory.GYM: ["gym"],
            FacilityCategory.FITNESS_CENTER: ["gym"],
            FacilityCategory.YOGA_STUDIO: ["gym"],
            FacilityCategory.PILATES_STUDIO: ["gym"],
            FacilityCategory.PERSONAL_TRAINING: ["gym"],
            FacilityCategory.CROSSFIT_GYM: ["gym"],
            FacilityCategory.MARTIAL_ARTS: ["gym"],
            FacilityCategory.DANCE_STUDIO: ["gym"],
            
            # Wellness
            FacilityCategory.WELLNESS_CENTER: ["spa", "health"],
            FacilityCategory.SPA: ["spa", "beauty_salon"],
            FacilityCategory.MASSAGE_THERAPY: ["spa", "health"],
            FacilityCategory.ACUPUNCTURE: ["health"],
            FacilityCategory.CHIROPRACTIC: ["health"],
            FacilityCategory.MEDITATION_CENTER: ["health"],
            FacilityCategory.HOLISTIC_HEALTH: ["health"],
            FacilityCategory.NATUROPATHY: ["health"],
            
            # Specialized
            FacilityCategory.NUTRITION_COUNSELING: ["health"],
            FacilityCategory.WEIGHT_LOSS_CENTER: ["health"],
            FacilityCategory.ADDICTION_RECOVERY: ["health"],
            FacilityCategory.SENIOR_FITNESS: ["gym", "health"],
            FacilityCategory.ADAPTIVE_FITNESS: ["gym", "health"]
        }
        
        # Search keywords for text search
        self.search_keywords = {
            # Healthcare keywords
            FacilityCategory.HOSPITAL: ["hospital", "medical center", "emergency room"],
            FacilityCategory.EMERGENCY_ROOM: ["emergency room", "emergency department", "ER"],
            FacilityCategory.URGENT_CARE: ["urgent care", "walk-in clinic", "immediate care"],
            FacilityCategory.MEDICAL_CLINIC: ["medical clinic", "family practice", "primary care"],
            FacilityCategory.SPECIALTY_CLINIC: ["specialty clinic", "specialist"],
            
            # Therapy keywords  
            FacilityCategory.PHYSICAL_THERAPY: ["physical therapy", "physiotherapy", "PT clinic"],
            FacilityCategory.OCCUPATIONAL_THERAPY: ["occupational therapy", "OT clinic"],
            FacilityCategory.SPEECH_THERAPY: ["speech therapy", "speech pathology"],
            FacilityCategory.REHABILITATION_CENTER: ["rehabilitation center", "rehab facility"],
            
            # Neurological keywords
            FacilityCategory.NEUROLOGY_CLINIC: ["neurology clinic", "neurologist"],
            FacilityCategory.NEUROTHERAPY_CLINIC: ["neurotherapy", "brain therapy"],
            FacilityCategory.BRAIN_INJURY_CENTER: ["brain injury center", "TBI center"],
            FacilityCategory.COGNITIVE_THERAPY: ["cognitive therapy", "memory clinic"],
            
            # Mental Health keywords
            FacilityCategory.MENTAL_HEALTH_CLINIC: ["mental health clinic", "behavioral health"],
            FacilityCategory.PSYCHOLOGY_PRACTICE: ["psychology practice", "psychologist"],
            FacilityCategory.PSYCHIATRY_CLINIC: ["psychiatry clinic", "psychiatrist"],
            FacilityCategory.COUNSELING_CENTER: ["counseling center", "therapy center"],
            FacilityCategory.SUPPORT_GROUP_CENTER: ["support group", "peer support"],
            
            # Fitness keywords
            FacilityCategory.GYM: ["gym", "fitness center", "health club"],
            FacilityCategory.FITNESS_CENTER: ["fitness center", "recreation center"],
            FacilityCategory.YOGA_STUDIO: ["yoga studio", "yoga center"],
            FacilityCategory.PILATES_STUDIO: ["pilates studio", "pilates center"],
            FacilityCategory.PERSONAL_TRAINING: ["personal training", "fitness coaching"],
            FacilityCategory.CROSSFIT_GYM: ["crossfit", "crossfit gym"],
            FacilityCategory.MARTIAL_ARTS: ["martial arts", "karate", "jiu jitsu", "boxing"],
            FacilityCategory.DANCE_STUDIO: ["dance studio", "dance center"],
            
            # Wellness keywords
            FacilityCategory.WELLNESS_CENTER: ["wellness center", "holistic health"],
            FacilityCategory.SPA: ["spa", "day spa", "wellness spa"],
            FacilityCategory.MASSAGE_THERAPY: ["massage therapy", "therapeutic massage"],
            FacilityCategory.ACUPUNCTURE: ["acupuncture", "acupuncturist"],
            FacilityCategory.CHIROPRACTIC: ["chiropractic", "chiropractor"],
            FacilityCategory.MEDITATION_CENTER: ["meditation center", "mindfulness center"],
            FacilityCategory.HOLISTIC_HEALTH: ["holistic health", "alternative medicine"],
            FacilityCategory.NATUROPATHY: ["naturopathy", "naturopathic doctor"],
            
            # Specialized keywords
            FacilityCategory.NUTRITION_COUNSELING: ["nutrition counseling", "dietitian"],
            FacilityCategory.WEIGHT_LOSS_CENTER: ["weight loss center", "bariatric center"],
            FacilityCategory.ADDICTION_RECOVERY: ["addiction recovery", "substance abuse treatment"],
            FacilityCategory.SENIOR_FITNESS: ["senior fitness", "senior center"],
            FacilityCategory.ADAPTIVE_FITNESS: ["adaptive fitness", "disability fitness"]
        }
    
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()
    
    async def search_facilities(self, criteria: SearchCriteria) -> Tuple[List[Facility], Optional[str]]:
        """Search for facilities based on comprehensive criteria - LIMITED TO 10 RESULTS"""
        
        try:
            # Get location coordinates
            location_coords, location_error = await self._geocode_location(criteria)
            if location_error:
                return [], location_error
            
            # Search for each facility category
            all_facilities = []
            for category in criteria.facility_categories:
                facilities = await self._search_category(category, location_coords, criteria)
                all_facilities.extend(facilities)
            
            # Process and filter results - LIMIT TO 10
            processed_facilities = self._process_facilities(all_facilities, location_coords, criteria)
            
            # Return only top 10 results as requested
            return processed_facilities[:10], None
            
        except Exception as e:
            logger.error(f"Facility search failed: {e}")
            return [], f"Search failed: {str(e)}"
    
    async def _geocode_location(self, criteria: SearchCriteria) -> Tuple[Optional[Dict], Optional[str]]:
        """Convert location to coordinates"""
        
        if criteria.location_type == LocationType.CURRENT_LOCATION and not criteria.location:
            return None, "Current location access not available. Please provide a location."
        
        try:
            geocode_url = f"{self.base_url}/geocode/json"
            params = {
                'address': criteria.location,
                'key': self.api_key
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.get(geocode_url, params=params, timeout=10)
            )
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return {
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'formatted_address': data['results'][0]['formatted_address']
                }, None
            else:
                return None, f"Location '{criteria.location}' not found."
                
        except Exception as e:
            return None, f"Location lookup failed: {str(e)}"
    
    async def _search_category(
        self, 
        category: FacilityCategory, 
        location_coords: Dict, 
        criteria: SearchCriteria
    ) -> List[Facility]:
        """Search for facilities of a specific category"""
        
        facilities = []
        
        # Try both API type search and text search
        facilities.extend(await self._nearby_search(category, location_coords, criteria))
        facilities.extend(await self._text_search(category, location_coords, criteria))
        
        return facilities
    
    async def _nearby_search(
        self, 
        category: FacilityCategory, 
        location_coords: Dict, 
        criteria: SearchCriteria
    ) -> List[Facility]:
        """Use Google Places Nearby Search API"""
        
        facilities = []
        api_types = self.api_type_mappings.get(category, ["health"])
        
        for api_type in api_types:
            try:
                nearby_url = f"{self.base_url}/place/nearbysearch/json"
                params = {
                    'location': f"{location_coords['lat']},{location_coords['lng']}",
                    'radius': min(criteria.radius_miles * 1609, 50000),  # Convert to meters, max 50km
                    'type': api_type,
                    'key': self.api_key
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(nearby_url, params=params, timeout=15)
                )
                
                data = response.json()
                
                if data['status'] == 'OK':
                    for place_data in data.get('results', []):
                        facility = self._parse_place_data(place_data, category, location_coords)
                        if facility and self._is_relevant_facility(facility, criteria):
                            facilities.append(facility)
                
                await asyncio.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Nearby search failed for {api_type}: {e}")
                continue
        
        return facilities
    
    async def _text_search(
        self, 
        category: FacilityCategory, 
        location_coords: Dict, 
        criteria: SearchCriteria
    ) -> List[Facility]:
        """Use Google Places Text Search API with targeted queries"""
        
        facilities = []
        keywords = self.search_keywords.get(category, [])
        
        # Make searches more specific to avoid hotels
        targeted_keywords = []
        for keyword in keywords[:2]:  # Limit to prevent too many API calls
            # Add specific qualifiers to avoid hotels
            if category in [FacilityCategory.WELLNESS_CENTER, FacilityCategory.SPA]:
                targeted_keywords.append(f"{keyword} wellness center")
                targeted_keywords.append(f"day {keyword}")  # "day spa" not hotel spa
            elif category == FacilityCategory.MASSAGE_THERAPY:
                targeted_keywords.append(f"massage therapy clinic")
                targeted_keywords.append(f"therapeutic massage center")
            elif category == FacilityCategory.MEDITATION_CENTER:
                targeted_keywords.append(f"meditation center")
                targeted_keywords.append(f"mindfulness studio")
            else:
                targeted_keywords.append(keyword)
        
        for keyword in targeted_keywords:
            try:
                text_search_url = f"{self.base_url}/place/textsearch/json"
                
                # Create more specific query
                search_query = f"{keyword} near {criteria.location}"
                
                # Add negative keywords to exclude hotels
                if category in [FacilityCategory.WELLNESS_CENTER, FacilityCategory.SPA, 
                               FacilityCategory.MASSAGE_THERAPY]:
                    search_query += " -hotel -resort -marriott -hilton"
                
                params = {
                    'query': search_query,
                    'location': f"{location_coords['lat']},{location_coords['lng']}",
                    'radius': min(criteria.radius_miles * 1609, 50000),
                    'key': self.api_key
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(text_search_url, params=params, timeout=15)
                )
                
                data = response.json()
                
                if data['status'] == 'OK':
                    for place_data in data.get('results', []):
                        facility = self._parse_place_data(place_data, category, location_coords)
                        if facility and self._is_relevant_facility(facility, criteria):
                            facilities.append(facility)
                            logger.debug(f"✅ Added facility from text search: {facility.name}")
                
                await asyncio.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Text search failed for {keyword}: {e}")
                continue
        
        return facilities
    
    def _parse_place_data(
        self, 
        place_data: Dict, 
        category: FacilityCategory, 
        location_coords: Dict
    ) -> Optional[Facility]:
        """Parse Google Places API response into Facility object with enhanced filtering"""
        
        try:
            # Basic information
            name = place_data.get('name', 'Unknown')
            place_id = place_data.get('place_id', '')
            
            # Pre-filter obviously irrelevant places by Google Places types
            place_types = place_data.get('types', [])
            
            # Exclude hotels, restaurants, and other irrelevant business types
            exclude_types = [
                'lodging', 'hotel', 'motel', 'resort', 'hostel',
                'restaurant', 'food', 'meal_takeaway', 'meal_delivery',
                'bar', 'night_club', 'liquor_store',
                'store', 'shopping_mall', 'department_store',
                'gas_station', 'car_dealer', 'car_rental', 'car_repair',
                'bank', 'atm', 'insurance_agency', 'real_estate_agency',
                'school', 'university', 'library',
                'church', 'cemetery', 'funeral_home',
                'tourist_attraction', 'amusement_park', 'zoo',
                'movie_theater', 'casino', 'bowling_alley'
            ]
            
            # If any exclude type is present, skip this place
            if any(exclude_type in place_types for exclude_type in exclude_types):
                logger.debug(f"Excluded by place type: {name} - {place_types}")
                return None
            
            # For wellness searches, require specific relevant types
            if category in [FacilityCategory.WELLNESS_CENTER, FacilityCategory.SPA, 
                           FacilityCategory.MASSAGE_THERAPY, FacilityCategory.MEDITATION_CENTER]:
                
                wellness_types = [
                    'spa', 'beauty_salon', 'hair_care', 'health', 'gym', 
                    'physiotherapist', 'doctor', 'establishment'
                ]
                
                # Must have at least one wellness-relevant type
                if not any(wellness_type in place_types for wellness_type in wellness_types):
                    logger.debug(f"Excluded wellness facility - no relevant types: {name} - {place_types}")
                    return None
            
            # Additional name-based pre-filtering
            name_lower = name.lower()
            immediate_exclude_names = [
                'marriott', 'hilton', 'hyatt', 'conrad', 'sheraton', 'westin', 
                'ritz', 'four seasons', 'intercontinental', 'doubletree',
                'holiday inn', 'best western', 'comfort inn', 'hampton inn',
                'mandarin oriental', 'the dominick', 'w new york'
            ]
            
            if any(exclude_name in name_lower for exclude_name in immediate_exclude_names):
                logger.debug(f"Excluded by name pattern: {name}")
                return None
            
            # Location
            geometry = place_data.get('geometry', {}).get('location', {})
            lat = geometry.get('lat', 0.0)
            lng = geometry.get('lng', 0.0)
            
            # Calculate distance
            distance = self._calculate_distance(
                location_coords['lat'], location_coords['lng'], lat, lng
            )
            
            # Address
            address = place_data.get('vicinity', place_data.get('formatted_address', ''))
            
            # Ratings
            rating = place_data.get('rating')
            review_count = place_data.get('user_ratings_total')
            
            # Business details
            business_status = place_data.get('business_status', 'OPERATIONAL')
            price_level = place_data.get('price_level')
            
            # Create maps URL
            maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            
            facility = Facility(
                name=name,
                category=category.value,
                address=address,
                latitude=lat,
                longitude=lng,
                distance_miles=distance,
                phone=place_data.get('formatted_phone_number'),
                rating=rating,
                review_count=review_count,
                business_status=business_status,
                price_level=price_level,
                maps_url=maps_url,
                place_id=place_id
            )
            
            logger.debug(f"✅ Successfully parsed facility: {name}")
            return facility
            
        except Exception as e:
            logger.warning(f"Failed to parse place data: {e}")
            return None
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in miles
        radius_miles = 3956
        
        return round(c * radius_miles, 1)
    
    def _is_relevant_facility(self, facility: Facility, criteria: SearchCriteria) -> bool:
        """Enhanced facility relevance checking with strict filtering"""
        
        facility_name = facility.name.lower()
        facility_address = facility.address.lower()
        
        # Comprehensive exclusion patterns for hotels and irrelevant businesses
        hotel_patterns = [
            'hotel', 'inn', 'resort', 'marriott', 'hilton', 'hyatt', 'conrad', 'sheraton',
            'westin', 'ritz', 'four seasons', 'intercontinental', 'doubletree', 'embassy',
            'courtyard', 'residence inn', 'extended stay', 'holiday inn', 'best western',
            'comfort inn', 'hampton inn', 'fairfield inn', 'springhill', 'homewood',
            'mandarin oriental', 'the dominick', 'w hotel', 'boutique hotel'
        ]
        
        restaurant_patterns = [
            'restaurant', 'cafe', 'bar', 'grill', 'bistro', 'diner', 'eatery',
            'kitchen', 'tavern', 'pub', 'steakhouse', 'pizzeria', 'bakery'
        ]
        
        retail_patterns = [
            'store', 'shop', 'mall', 'market', 'pharmacy', 'cvs', 'walgreens',
            'target', 'walmart', 'costco', 'shopping', 'retail', 'outlet'
        ]
        
        other_exclude_patterns = [
            'bank', 'atm', 'gas station', 'school', 'university', 'college',
            'church', 'temple', 'mosque', 'synagogue', 'library', 'museum',
            'theater', 'cinema', 'entertainment', 'nightclub', 'casino',
            'car wash', 'auto repair', 'mechanic', 'insurance', 'real estate'
        ]
        
        # Combine all exclusion patterns
        all_exclude_patterns = hotel_patterns + restaurant_patterns + retail_patterns + other_exclude_patterns
        
        # Check if facility name contains any exclusion patterns
        if any(pattern in facility_name for pattern in all_exclude_patterns):
            logger.debug(f"Excluded facility by name pattern: {facility.name}")
            return False
        
        # Check if address contains hotel/retail indicators
        address_exclude_patterns = ['hotel', 'mall', 'shopping center', 'plaza hotel']
        if any(pattern in facility_address for pattern in address_exclude_patterns):
            logger.debug(f"Excluded facility by address pattern: {facility.name}")
            return False
        
        # Positive filtering - facility must contain relevant health/wellness keywords
        health_wellness_keywords = [
            # Healthcare
            'hospital', 'medical', 'clinic', 'doctor', 'physician', 'health',
            'emergency', 'urgent care', 'rehabilitation', 'therapy', 'treatment',
            'psychiatric', 'psychology', 'counseling', 'mental health',
            
            # Therapy & Rehabilitation  
            'physical therapy', 'occupational therapy', 'speech therapy',
            'physiotherapy', 'rehab', 'rehabilitation', 'recovery',
            
            # Wellness & Fitness
            'wellness', 'fitness', 'gym', 'yoga', 'pilates', 'meditation',
            'spa', 'massage', 'acupuncture', 'chiropractic', 'holistic',
            'nutrition', 'naturopathy', 'therapeutic', 'healing',
            
            # Mental Health
            'psychiatry', 'psychology', 'behavioral health', 'counseling',
            'therapy', 'mental health', 'addiction', 'recovery', 'support group'
        ]
        
        # Facility must contain at least one relevant keyword
        has_relevant_keyword = any(keyword in facility_name for keyword in health_wellness_keywords)
        
        if not has_relevant_keyword:
            logger.debug(f"Excluded facility - no relevant keywords: {facility.name}")
            return False
        
        # Distance constraint
        if facility.distance_miles and facility.distance_miles > criteria.radius_miles:
            logger.debug(f"Excluded facility - too far: {facility.name} ({facility.distance_miles} miles)")
            return False
        
        # Business status check
        if facility.business_status not in ['OPERATIONAL', 'OPEN']:
            logger.debug(f"Excluded facility - not operational: {facility.name}")
            return False
        
        logger.debug(f"✅ Facility passed all filters: {facility.name}")
        return True
    
    def _process_facilities(
        self, 
        facilities: List[Facility], 
        location_coords: Dict, 
        criteria: SearchCriteria
    ) -> List[Facility]:
        """Process and sort facilities"""
        
        # Remove duplicates
        unique_facilities = self._deduplicate_facilities(facilities)
        
        # Sort by relevance
        sorted_facilities = self._sort_facilities(unique_facilities, criteria)
        
        return sorted_facilities
    
    def _deduplicate_facilities(self, facilities: List[Facility]) -> List[Facility]:
        """Remove duplicate facilities"""
        
        seen = set()
        unique_facilities = []
        
        for facility in facilities:
            # Create unique key based on name and location
            key = (
                facility.name.lower().strip(),
                round(facility.latitude, 4),
                round(facility.longitude, 4)
            )
            
            if key not in seen:
                seen.add(key)
                unique_facilities.append(facility)
        
        return unique_facilities
    
    def _sort_facilities(self, facilities: List[Facility], criteria: SearchCriteria) -> List[Facility]:
        """Sort facilities by relevance and quality"""
        
        def sort_key(facility):
            # Review count (primary factor for reliability)
            review_score = min((facility.review_count or 0) / 100, 5.0)
            
            # Rating score
            rating_score = facility.rating or 0
            
            # Distance score (closer is better)
            distance_score = max(0, 5 - (facility.distance_miles or 0) / 5)
            
            # Business status bonus
            status_bonus = 1 if facility.business_status == 'OPERATIONAL' else 0
            
            # Combined score
            total_score = (
                review_score * 0.4 +
                rating_score * 0.3 +
                distance_score * 0.2 +
                status_bonus * 0.1
            )
            
            return total_score
        
        return sorted(facilities, key=sort_key, reverse=True)

class ResponseFormatter:
    """Formats search results into user-friendly responses with JSON support - FIXED VERSION"""
    
    def format_response(
        self, 
        facilities: List[Facility], 
        criteria: SearchCriteria, 
        processing_time: float
    ) -> Dict[str, Any]:
        """Format comprehensive response with both text and structured JSON data - FIXED"""
        
        if not facilities:
            return self._format_no_results_response(criteria, processing_time)
        
        # IMPORTANT: Keep facilities as Facility objects for text generation
        # Generate text response first while facilities are still objects
        response_text = self._generate_response_text(facilities, criteria)
        
        # Generate structured JSON data for frontend rendering
        structured_data = self._generate_structured_response(facilities, criteria)
        
        # Convert facilities to dict format AFTER text generation
        facilities_dict = [facility.to_dict() for facility in facilities]
        
        return {
            "success": True,
            "query": criteria.original_query,
            "location": criteria.location,
            "categories_searched": [cat.value for cat in criteria.facility_categories],
            "total_results": len(facilities),
            "urgency_level": criteria.urgency.value,
            "search_radius_miles": criteria.radius_miles,
            
            # Text response for synthesis/display
            "response": response_text,
            
            # Structured data for frontend rendering
            "structured_data": structured_data,
            
            # Facility list for compatibility
            "facilities": facilities_dict,
            
            "processing_time": processing_time,
            "confidence": criteria.confidence,
            
            # Frontend rendering flag
            "render_mode": "structured"  # Indicates frontend should use structured_data
        }
    
    def _generate_structured_response(self, facilities: List[Facility], criteria: SearchCriteria) -> Dict[str, Any]:
        """Generate structured data optimized for frontend rendering"""
        
        # Create header information
        categories_display = []
        for cat in criteria.facility_categories:
            categories_display.append({
                "id": cat.value,
                "name": cat.value.replace('_', ' ').title(),
                "icon": self._get_category_icon(cat)
            })
        
        # Process facilities for frontend - LIMIT TO 10 AS REQUESTED
        processed_facilities = []
        for i, facility in enumerate(facilities[:10]):  # Limit to top 10
            facility_data = {
                "id": facility.place_id or f"facility_{i}",
                "name": facility.name,
                "category": {
                    "id": facility.category,
                    "name": facility.category.replace('_', ' ').title(),
                    "icon": self._get_category_icon_by_value(facility.category)
                },
                "location": {
                    "address": facility.address,
                    "latitude": facility.latitude,
                    "longitude": facility.longitude,
                    "distance_miles": facility.distance_miles,
                    "distance_display": f"{facility.distance_miles} miles away" if facility.distance_miles else None
                },
                "contact": {
                    "phone": facility.phone,
                    "website": facility.website,
                    "email": facility.email
                },
                "reviews": {
                    "rating": facility.rating,
                    "review_count": facility.review_count,
                    "stars_display": "⭐" * int(facility.rating) if facility.rating else None,
                    "rating_display": f"{facility.rating}/5" if facility.rating else None
                },
                "business_info": {
                    "status": facility.business_status,
                    "price_level": facility.price_level,
                    "price_display": "💰" * facility.price_level if facility.price_level else None,
                    "hours": facility.hours
                },
                "actions": {
                    "maps_url": facility.maps_url,
                    "booking_url": facility.booking_url,
                    "call_action": f"tel:{facility.phone}" if facility.phone else None
                },
                "accessibility": facility.accessibility_features,
                "services": facility.services,
                "insurance": facility.insurance_accepted
            }
            processed_facilities.append(facility_data)
        
        # Create search metadata
        search_metadata = {
            "original_query": criteria.original_query,
            "parsed_location": criteria.location,
            "search_radius": criteria.radius_miles,
            "urgency_level": {
                "id": criteria.urgency.value,
                "name": criteria.urgency.value.replace('_', ' ').title(),
                "color": self._get_urgency_color(criteria.urgency)
            },
            "total_found": len(facilities),
            "showing_count": len(processed_facilities),
            "has_more": len(facilities) > len(processed_facilities)
        }
        
        # Urgency-specific messages
        urgency_messages = self._get_urgency_messages(criteria.urgency)
        
        return {
            "header": {
                "title": f"Found {len(facilities)} facilities near {criteria.location}",
                "subtitle": f"Showing {categories_display[0]['name'].lower()} results" if len(categories_display) == 1 else "Multiple facility types",
                "categories": categories_display
            },
            "search_metadata": search_metadata,
            "facilities": processed_facilities,
            "urgency_info": urgency_messages,
            "rendering_hints": {
                "map_view_available": True,
                "list_view_default": True,
                "filter_options": ["distance", "rating", "price_level"],
                "sort_options": ["relevance", "distance", "rating", "review_count"]
            }
        }
    
    def _get_category_icon(self, category: FacilityCategory) -> str:
        """Get appropriate icon for facility category"""
        icon_mapping = {
            # Healthcare
            FacilityCategory.HOSPITAL: "🏥",
            FacilityCategory.EMERGENCY_ROOM: "🚑",
            FacilityCategory.URGENT_CARE: "⚕️",
            FacilityCategory.MEDICAL_CLINIC: "🩺",
            FacilityCategory.SPECIALTY_CLINIC: "👨‍⚕️",
            
            # Therapy
            FacilityCategory.PHYSICAL_THERAPY: "🏃‍♂️",
            FacilityCategory.OCCUPATIONAL_THERAPY: "🖐️",
            FacilityCategory.SPEECH_THERAPY: "🗣️",
            FacilityCategory.REHABILITATION_CENTER: "🏥",
            
            # Neurological
            FacilityCategory.NEUROLOGY_CLINIC: "🧠",
            FacilityCategory.BRAIN_INJURY_CENTER: "🧠",
            FacilityCategory.COGNITIVE_THERAPY: "🧠",
            
            # Mental Health
            FacilityCategory.MENTAL_HEALTH_CLINIC: "🧘‍♀️",
            FacilityCategory.PSYCHOLOGY_PRACTICE: "💭",
            FacilityCategory.COUNSELING_CENTER: "🗨️",
            
            # Fitness
            FacilityCategory.GYM: "💪",
            FacilityCategory.FITNESS_CENTER: "🏋️‍♂️",
            FacilityCategory.YOGA_STUDIO: "🧘‍♀️",
            FacilityCategory.PILATES_STUDIO: "🤸‍♀️",
            FacilityCategory.MARTIAL_ARTS: "🥋",
            FacilityCategory.DANCE_STUDIO: "💃",
            
            # Wellness
            FacilityCategory.WELLNESS_CENTER: "🌿",
            FacilityCategory.SPA: "💆‍♀️",
            FacilityCategory.MASSAGE_THERAPY: "💆",
            FacilityCategory.MEDITATION_CENTER: "🧘",
            FacilityCategory.ACUPUNCTURE: "🏮",
            FacilityCategory.CHIROPRACTIC: "🦴"
        }
        return icon_mapping.get(category, "🏢")
    
    def _get_category_icon_by_value(self, category_value: str) -> str:
        """Get icon by category value string"""
        try:
            category = FacilityCategory(category_value)
            return self._get_category_icon(category)
        except ValueError:
            return "🏢"
    
    def _get_urgency_color(self, urgency: UrgencyLevel) -> str:
        """Get color code for urgency level"""
        color_mapping = {
            UrgencyLevel.EMERGENCY: "#DC2626",  # Red
            UrgencyLevel.URGENT: "#F59E0B",     # Amber
            UrgencyLevel.ROUTINE: "#10B981",    # Green
            UrgencyLevel.WELLNESS: "#6366F1"    # Indigo
        }
        return color_mapping.get(urgency, "#6B7280")  # Gray default
    
    def _get_urgency_messages(self, urgency: UrgencyLevel) -> Dict[str, Any]:
        """Get urgency-specific messages and actions"""
        messages = {
            UrgencyLevel.EMERGENCY: {
                "message": "🚨 For medical emergencies, call 911 immediately",
                "action_text": "Call 911",
                "action_url": "tel:911",
                "priority": "critical"
            },
            UrgencyLevel.URGENT: {
                "message": "⚡ For urgent needs, call ahead to check availability",
                "action_text": "Call facility",
                "priority": "high"
            },
            UrgencyLevel.ROUTINE: {
                "message": "📅 Schedule an appointment for routine care",
                "action_text": "Schedule appointment",
                "priority": "normal"
            },
            UrgencyLevel.WELLNESS: {
                "message": "🌿 Take time to explore your wellness options",
                "action_text": "Learn more",
                "priority": "low"
            }
        }
        return messages.get(urgency, {"message": "", "priority": "normal"})
    
    def _generate_response_text(self, facilities: List[Facility], criteria: SearchCriteria) -> str:
        """Generate human-readable response text (for synthesis purposes) - FIXED VERSION"""
        
        # Create header
        categories_str = ', '.join([
            cat.value.replace('_', ' ').title() 
            for cat in criteria.facility_categories
        ])
        
        location_str = criteria.location or "your area"
        
        response_parts = [
            f"I found {len(facilities)} {categories_str.lower()} near {location_str}:",
            ""
        ]
        
        # Add facility details - LIMIT TO 10 AS REQUESTED
        for i, facility in enumerate(facilities[:10], 1):  # Show top 10 only
            facility_parts = [f"**{i}. {facility.name}**"]
            
            if facility.address:
                facility_parts.append(f"   📍 {facility.address}")
            
            if facility.phone:
                facility_parts.append(f"   📞 {facility.phone}")
            
            # Show review count and rating
            if facility.review_count and facility.rating:
                stars = "⭐" * int(facility.rating)
                facility_parts.append(f"   👥 {facility.review_count} reviews | {stars} {facility.rating}/5")
            elif facility.review_count:
                facility_parts.append(f"   👥 {facility.review_count} reviews")
            elif facility.rating:
                stars = "⭐" * int(facility.rating)
                facility_parts.append(f"   {stars} {facility.rating}/5")
            
            if facility.distance_miles:
                facility_parts.append(f"   📏 {facility.distance_miles} miles away")
            
            if facility.price_level:
                price_symbols = "💰" * facility.price_level
                facility_parts.append(f"   {price_symbols} Price level: {facility.price_level}/4")
            
            facility_parts.append(f"   🗺️ [View on Maps]({facility.maps_url})")
            facility_parts.append("")
            
            response_parts.extend(facility_parts)
        
        # Add urgency-specific advice
        if criteria.urgency == UrgencyLevel.EMERGENCY:
            response_parts.extend([
                "",
                "🚨 **For medical emergencies, call 911 immediately.**"
            ])
        elif criteria.urgency == UrgencyLevel.URGENT:
            response_parts.extend([
                "",
                "⚡ **For urgent needs, call ahead to check availability.**"
            ])
        
        return "\n".join(response_parts)
    
    def _format_no_results_response(self, criteria: SearchCriteria, processing_time: float) -> Dict[str, Any]:
        """Format response when no results found"""
        
        categories_str = ', '.join([
            cat.value.replace('_', ' ') 
            for cat in criteria.facility_categories
        ])
        
        response_text = f"""I couldn't find any {categories_str} near {criteria.location or 'your location'} within {criteria.radius_miles} miles.

**Suggestions:**
• Expand your search radius
• Try nearby cities or areas
• Search for more general facility types
• Check online directories
• Contact your insurance provider for covered facilities"""
        
        # Structured data for no results
        structured_data = {
            "header": {
                "title": f"No {categories_str} found",
                "subtitle": f"Near {criteria.location or 'your location'}"
            },
            "search_metadata": {
                "original_query": criteria.original_query,
                "parsed_location": criteria.location,
                "search_radius": criteria.radius_miles,
                "total_found": 0
            },
            "suggestions": [
                "Expand your search radius",
                "Try nearby cities or areas", 
                "Search for more general facility types",
                "Check online directories",
                "Contact your insurance provider"
            ],
            "facilities": []
        }
        
        return {
            "success": True,
            "query": criteria.original_query,
            "total_results": 0,
            "response": response_text,
            "structured_data": structured_data,
            "facilities": [],
            "processing_time": processing_time,
            "render_mode": "no_results"
        }

class HealthcareWellnessLocator:
    """Main Healthcare & Wellness Locator Agent with JSON response support - FIXED VERSION"""
    
    def __init__(self, gemini_api_key: str = None, google_places_api_key: str = None):
        """Initialize the comprehensive locator agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.google_places_api_key = google_places_api_key or os.getenv("GOOGLE_PLACES_API_KEY")
        
        self._validate_configuration()
        self._initialize_components()
        
        logger.info("✅ Healthcare & Wellness Locator Agent initialized with JSON support")
    
    def _validate_configuration(self):
        """Validate required configuration"""
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        if not self.google_places_api_key:
            raise ValueError("Google Places API key is required")
    
    def _initialize_components(self):
        """Initialize all components using dependency injection pattern"""
        try:
            # Initialize Gemini
            genai.configure(api_key=self.gemini_api_key)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize components
            self.parser = LLMSearchParser(gemini_model)
            self.connector = PlacesAPIConnector(self.google_places_api_key)
            self.formatter = ResponseFormatter()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def find_facilities(self, query: str) -> Dict[str, Any]:
        """Main method to find healthcare and wellness facilities with JSON support - FIXED VERSION"""
        
        start_time = datetime.now()
        
        try:
            # Parse query using advanced LLM intelligence
            logger.info(f"🧠 Parsing query: {query[:50]}...")
            criteria = await self.parser.parse_query(query)
            
            logger.info(f"🎯 Parsed - Categories: {[c.value for c in criteria.facility_categories]}")
            logger.info(f"📍 Location: {criteria.location}, Radius: {criteria.radius_miles} miles")
            
            # Search for facilities
            logger.info("🔍 Searching for facilities...")
            facilities, error = await self.connector.search_facilities(criteria)
            
            if error:
                return self._create_error_response(error, start_time)
            
            # Format response with both text and JSON
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # IMPORTANT: Pass Facility objects to formatter, not dictionaries
            response = self.formatter.format_response(facilities, criteria, processing_time)
            
            logger.info(f"✅ Found {len(facilities)} facilities in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.exception(f"❌ Error finding facilities: {e}")
            return self._create_error_response(f"Search failed: {str(e)}", start_time)
    
    def _create_error_response(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error response with JSON support"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": False,
            "error": error_message,
            "response": f"I encountered an issue while searching: {error_message}",
            "structured_data": {
                "header": {
                    "title": "Search Error",
                    "subtitle": "Unable to complete facility search"
                },
                "error": {
                    "message": error_message,
                    "suggestions": [
                        "Check your internet connection",
                        "Try a different location",
                        "Simplify your search terms"
                    ]
                }
            },
            "facilities": [],
            "processing_time": processing_time,
            "render_mode": "error"
        }

# Testing Interface
async def test_enhanced_locator():
    """Enhanced testing interface with JSON output examples"""
    
    print("=" * 100)
    print("🏥 FIXED HEALTHCARE & WELLNESS LOCATOR - LIMITED TO 10 RESULTS")
    print("=" * 100)
    
    # Check API keys
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    places_key = os.getenv("GOOGLE_PLACES_API_KEY")
    
    if not gemini_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return
    if not places_key:
        print("❌ GOOGLE_PLACES_API_KEY not found in environment variables")
        return
    
    # Initialize agent
    print("🚀 Initializing Fixed Healthcare & Wellness Locator...")
    try:
        locator = HealthcareWellnessLocator(gemini_key, places_key)
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("\n🔧 **FIXES APPLIED:**")
    print("   • Fixed AttributeError: 'dict' object has no attribute 'name'")
    print("   • Limited results to 10 locations as requested")
    print("   • Improved error handling and data flow")
    print("   • Maintained both text and JSON response formats")
    
    print("\n💡 **COMMANDS:**")
    print("   • Type your search query naturally")
    print("   • 'demo' - Run comprehensive demo queries")
    print("   • 'quit' - Exit")
    print("=" * 100)
    
    while True:
        try:
            user_input = input("\n🔍 Enter your facility search: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n👋 Thank you for testing the Fixed Locator!")
                break
            
            elif user_input.lower() == 'demo':
                demo_queries = [
                    "yoga studios in Boston",
                    "TBI rehabilitation centers near 90210", 
                    "wellness spas and massage therapy in NYC"
                ]
                
                for query in demo_queries:
                    print(f"\n🧪 **Demo Query:** '{query}'")
                    result = await locator.find_facilities(query)
                    print(f"**Result:** {result['success']} - {len(result.get('facilities', []))} facilities (max 10)")
                    print(f"**Render Mode:** {result.get('render_mode', 'standard')}")
                    
                    if result['success'] and result.get('facilities'):
                        print(f"**Sample Facilities:**")
                        for i, facility in enumerate(result['facilities'][:3]):  # Show first 3
                            print(f"  {i+1}. {facility['name']} - {facility.get('address', 'N/A')}")
                continue
            
            # Process the query
            print(f"\n🤖 Processing: '{user_input}'...")
            
            result = await locator.find_facilities(user_input)
            
            if result["success"]:
                print(f"\n📊 **Results Summary:**")
                print(f"   🏢 Total Found: {result['total_results']} (showing max 10)")
                print(f"   📍 Location: {result.get('location', 'N/A')}")
                print(f"   🏥 Categories: {', '.join(result.get('categories_searched', []))}")
                print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
                print(f"   🎨 Render Mode: {result.get('render_mode', 'standard')}")
                
                if result.get('facilities'):
                    print(f"\n🏥 **Top Facilities Found:**")
                    for i, facility in enumerate(result['facilities'][:5], 1):  # Show top 5
                        print(f"  {i}. {facility['name']}")
                        if facility.get('address'):
                            print(f"     📍 {facility['address']}")
                        if facility.get('rating'):
                            print(f"     ⭐ {facility['rating']}/5 ({facility.get('review_count', 0)} reviews)")
                        if facility.get('distance_miles'):
                            print(f"     📏 {facility['distance_miles']} miles away")
                        print()
                
            else:
                print(f"\n❌ **Error:** {result['error']}")
                print(f"**Response:** {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Fixed Healthcare & Wellness Locator Agent...")
    asyncio.run(test_enhanced_locator())