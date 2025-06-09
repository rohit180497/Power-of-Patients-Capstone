import os
import json
import logging
import asyncio
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re
import urllib.parse

logger = logging.getLogger(__name__)

class FacilityType(Enum):
    """Standardized facility types for healthcare locator"""
    HOSPITAL = "hospital"
    REHABILITATION_CENTER = "rehabilitation_center"
    NEUROTHERAPY_CLINIC = "neurotherapy_clinic"
    PHYSICAL_THERAPY = "physical_therapy"
    MENTAL_HEALTH = "mental_health_clinic"
    URGENT_CARE = "urgent_care"
    EMERGENCY_ROOM = "emergency_room"
    NEUROLOGY_CLINIC = "neurology_clinic"
    BRAIN_INJURY_CENTER = "brain_injury_center"
    SPEECH_THERAPY = "speech_therapy"
    OCCUPATIONAL_THERAPY = "occupational_therapy"
    GENERAL_CLINIC = "clinic"

@dataclass
class LocationQuery:
    """Parsed location query structure"""
    original_query: str
    facility_types: List[FacilityType]
    location: str
    location_type: str  # "zip_code", "city", "address", "current_location"
    urgency: str  # "emergency", "urgent", "routine"
    radius_miles: int
    additional_requirements: List[str]
    confidence: float
    reasoning: str

@dataclass
class HealthcareFacility:
    """Healthcare facility information structure"""
    name: str
    address: str
    phone: Optional[str]
    rating: Optional[float]
    rating_count: Optional[int]
    place_id: str
    latitude: float
    longitude: float
    facility_type: str
    business_status: str
    maps_url: str
    website: Optional[str] = None
    hours: Optional[Dict] = None
    distance_miles: Optional[float] = None

class IntelligentLocationParser:
    """LLM-powered location and facility type parser"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Facility type mappings for Google Places API
        self.facility_mappings = {
            FacilityType.HOSPITAL: ["hospital", "medical_center"],
            FacilityType.REHABILITATION_CENTER: ["physiotherapist", "physical_therapy", "rehabilitation"],
            FacilityType.NEUROTHERAPY_CLINIC: ["doctor", "clinic", "neurologist"],
            FacilityType.PHYSICAL_THERAPY: ["physiotherapist", "physical_therapy"],
            FacilityType.MENTAL_HEALTH: ["psychologist", "psychiatrist", "mental_health"],
            FacilityType.URGENT_CARE: ["urgent_care", "walk_in_clinic"],
            FacilityType.EMERGENCY_ROOM: ["hospital", "emergency_room"],
            FacilityType.NEUROLOGY_CLINIC: ["doctor", "neurologist", "clinic"],
            FacilityType.BRAIN_INJURY_CENTER: ["rehabilitation", "clinic", "doctor"],
            FacilityType.SPEECH_THERAPY: ["speech_therapist", "clinic"],
            FacilityType.OCCUPATIONAL_THERAPY: ["occupational_therapist", "rehabilitation"],
            FacilityType.GENERAL_CLINIC: ["clinic", "doctor"]
        }
    
    async def parse_location_query(self, query: str) -> LocationQuery:
        """Parse natural language query for location and facility information"""
        
        parsing_prompt = f"""
You are an expert healthcare location query parser. Extract location and facility information from the user's natural language query.

USER QUERY: "{query}"

TASK: Parse this query to extract:
1. What type of healthcare facility they need
2. Where they want to search (location)
3. Any specific requirements or urgency

FACILITY TYPES (choose from these):
- "hospital" - General hospitals, medical centers
- "rehabilitation_center" - Physical rehabilitation, recovery centers
- "neurotherapy_clinic" - Neurology clinics, brain treatment centers
- "physical_therapy" - Physical therapy clinics, physiotherapy
- "mental_health" - Mental health clinics, psychology, psychiatry
- "urgent_care" - Urgent care centers, walk-in clinics
- "emergency_room" - Emergency departments, ER
- "neurology_clinic" - Neurology specialists, neurologists
- "brain_injury_center" - TBI treatment, brain injury rehabilitation
- "speech_therapy" - Speech therapy, speech pathology
- "occupational_therapy" - Occupational therapy clinics
- "general_clinic" - General medical clinics, family practice

LOCATION TYPES:
- "zip_code" - 5-digit ZIP code (e.g., "90210")
- "city" - City name with or without state (e.g., "Boston MA", "Boston")
- "address" - Full street address
- "current_location" - User wants to use their current location ("near me", "nearby")

URGENCY LEVELS:
- "emergency" - Immediate medical attention needed
- "urgent" - Same day or within 24 hours
- "routine" - General appointment scheduling

DEFAULT VALUES:
- If no specific location mentioned, assume "current_location"
- If no urgency specified, assume "routine"
- Default search radius: 10 miles for routine, 25 miles for urgent, 50 miles for emergency

EXAMPLES:
- "hospitals near 02134" → hospital, zip_code: 02134
- "TBI rehabilitation centers in Boston" → brain_injury_center, city: Boston
- "urgent care near me" → urgent_care, current_location
- "neurologists in Los Angeles CA" → neurology_clinic, city: Los Angeles CA

Respond in JSON format:
{{
    "facility_types": ["list", "of", "facility", "types"],
    "location": "extracted location string",
    "location_type": "zip_code/city/address/current_location",
    "urgency": "emergency/urgent/routine",
    "radius_miles": number,
    "additional_requirements": ["any", "specific", "requirements"],
    "confidence": 0.0-1.0,
    "reasoning": "explanation of parsing decisions"
}}

JSON Response:"""

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(
                    parsing_prompt,
                    safety_settings=self.safety_settings
                )
            )
            
            result_text = response.text.strip()
            # Clean JSON response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
            
            result = json.loads(result_text)
            
            # Convert facility type strings to enums
            facility_types = []
            for facility_str in result.get("facility_types", []):
                try:
                    facility_types.append(FacilityType(facility_str))
                except ValueError:
                    logger.warning(f"Unknown facility type: {facility_str}")
                    facility_types.append(FacilityType.GENERAL_CLINIC)
            
            if not facility_types:
                facility_types = [FacilityType.GENERAL_CLINIC]
            
            return LocationQuery(
                original_query=query,
                facility_types=facility_types,
                location=result.get("location", ""),
                location_type=result.get("location_type", "current_location"),
                urgency=result.get("urgency", "routine"),
                radius_miles=result.get("radius_miles", 10),
                additional_requirements=result.get("additional_requirements", []),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"Location query parsing failed: {e}")
            return self._fallback_parsing(query)
    
    def _fallback_parsing(self, query: str) -> LocationQuery:
        """Fallback parsing when LLM fails"""
        query_lower = query.lower()
        
        # Simple keyword detection
        facility_type = FacilityType.GENERAL_CLINIC
        if any(word in query_lower for word in ['hospital', 'emergency', 'er']):
            facility_type = FacilityType.HOSPITAL
        elif any(word in query_lower for word in ['rehab', 'rehabilitation', 'physical therapy']):
            facility_type = FacilityType.REHABILITATION_CENTER
        elif any(word in query_lower for word in ['urgent care', 'urgent']):
            facility_type = FacilityType.URGENT_CARE
        elif any(word in query_lower for word in ['neuro', 'brain', 'tbi']):
            facility_type = FacilityType.NEUROLOGY_CLINIC
        
        # Simple location detection
        zip_match = re.search(r'\b\d{5}\b', query)
        location = zip_match.group() if zip_match else ""
        location_type = "zip_code" if zip_match else "current_location"
        
        urgency = "urgent" if "urgent" in query_lower else "routine"
        
        return LocationQuery(
            original_query=query,
            facility_types=[facility_type],
            location=location,
            location_type=location_type,
            urgency=urgency,
            radius_miles=15,
            additional_requirements=[],
            confidence=0.3,
            reasoning="Fallback parsing due to LLM error"
        )

class GooglePlacesConnector:
    """Google Places API connector for healthcare facility search"""
    
    def __init__(self, api_key: str):
        self.api_key = os.getenv("GOOGLE_PLACES_API_KEY")
        self.base_url = "https://maps.googleapis.com/maps/api"
        
        # Session for connection pooling
        self.session = requests.Session()
        
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()
    
    async def search_facilities(
        self, 
        location_query: LocationQuery
    ) -> Tuple[List[HealthcareFacility], Optional[str]]:
        """Search for healthcare facilities using Google Places API"""
        
        try:
            # Step 1: Get location coordinates
            location_coords, location_error = await self._get_location_coordinates(location_query)
            if location_error:
                return [], location_error
            
            # Step 2: Search for each facility type
            all_facilities = []
            
            for facility_type in location_query.facility_types:
                facilities = await self._search_facility_type(
                    facility_type, location_coords, location_query.radius_miles
                )
                all_facilities.extend(facilities)
            
            # Step 3: Remove duplicates and sort by rating/distance
            unique_facilities = self._deduplicate_facilities(all_facilities)
            sorted_facilities = self._sort_facilities(unique_facilities, location_coords)
            
            # Step 4: Add distance calculations
            facilities_with_distance = self._add_distance_info(sorted_facilities, location_coords)
            
            return facilities_with_distance[:15], None  # Return top 15 results
            
        except Exception as e:
            logger.error(f"Facility search failed: {e}")
            return [], f"Search failed: {str(e)}"
    
    async def _get_location_coordinates(self, location_query: LocationQuery) -> Tuple[Optional[Dict], Optional[str]]:
        """Get latitude/longitude for the search location"""
        
        if location_query.location_type == "current_location" and not location_query.location:
            return None, "Current location access not available. Please provide a ZIP code or city name."
        
        # Geocoding API call
        geocode_url = f"{self.base_url}/geocode/json"
        params = {
            'address': location_query.location,
            'key': self.api_key
        }
        
        try:
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
                return None, f"Location '{location_query.location}' not found. Please check the spelling or try a different format."
                
        except Exception as e:
            return None, f"Location lookup failed: {str(e)}"
    
    async def _search_facility_type(
        self, 
        facility_type: FacilityType, 
        location_coords: Dict, 
        radius_miles: int
    ) -> List[HealthcareFacility]:
        """Search for a specific facility type"""
        
        if not location_coords:
            return []
        
        # Convert miles to meters for Google API
        radius_meters = int(radius_miles * 1609.34)
        
        facilities = []
        
        # For hospitals and emergency rooms, use both type search and text search
        if facility_type in [FacilityType.HOSPITAL, FacilityType.EMERGENCY_ROOM]:
            # Text search for hospitals
            facilities.extend(await self._text_search_hospitals(location_coords, radius_meters))
        
        # Get search types for this facility
        search_types = self._get_search_types_for_facility(facility_type)
        
        for search_type in search_types:
            try:
                # Nearby Search API
                nearby_url = f"{self.base_url}/place/nearbysearch/json"
                params = {
                    'location': f"{location_coords['lat']},{location_coords['lng']}",
                    'radius': min(radius_meters, 50000),  # Max 50km for API
                    'type': search_type,
                    'key': self.api_key
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(nearby_url, params=params, timeout=15)
                )
                
                data = response.json()
                
                if data['status'] == 'OK':
                    for place in data.get('results', []):
                        facility = self._parse_place_data(place, facility_type.value)
                        if facility:
                            facilities.append(facility)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Search failed for {search_type}: {e}")
                continue
        
        return facilities
    
    async def _text_search_hospitals(self, location_coords: Dict, radius_meters: int) -> List[HealthcareFacility]:
        """Use text search specifically for hospitals"""
        
        facilities = []
        
        # Search queries specifically for hospitals
        hospital_queries = [
            "hospital",
            "emergency room",
            "medical center",
            "general hospital"
        ]
        
        for query in hospital_queries:
            try:
                # Text Search API
                text_search_url = f"{self.base_url}/place/textsearch/json"
                params = {
                    'query': query,
                    'location': f"{location_coords['lat']},{location_coords['lng']}",
                    'radius': min(radius_meters, 50000),
                    'key': self.api_key
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.session.get(text_search_url, params=params, timeout=15)
                )
                
                data = response.json()
                
                if data['status'] == 'OK':
                    for place in data.get('results', []):
                        facility = self._parse_place_data(place, "hospital")
                        if facility:
                            facilities.append(facility)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Text search failed for {query}: {e}")
                continue
        
        return facilities
    
    def _get_search_types_for_facility(self, facility_type: FacilityType) -> List[str]:
        """Get Google Places API search types for facility"""
        
        # Google Places API types - more specific to avoid hotels/irrelevant results
        type_mappings = {
            FacilityType.HOSPITAL: ["hospital"],
            FacilityType.REHABILITATION_CENTER: ["physiotherapist"],
            FacilityType.NEUROTHERAPY_CLINIC: ["doctor"],
            FacilityType.PHYSICAL_THERAPY: ["physiotherapist"],
            FacilityType.MENTAL_HEALTH: ["doctor"],
            FacilityType.URGENT_CARE: ["doctor"],
            FacilityType.EMERGENCY_ROOM: ["hospital"],
            FacilityType.NEUROLOGY_CLINIC: ["doctor"],
            FacilityType.BRAIN_INJURY_CENTER: ["doctor"],
            FacilityType.SPEECH_THERAPY: ["doctor"],
            FacilityType.OCCUPATIONAL_THERAPY: ["physiotherapist"],
            FacilityType.GENERAL_CLINIC: ["doctor"]
        }
        
        return type_mappings.get(facility_type, ["doctor"])
    
    def _parse_place_data(self, place: Dict, facility_type: str) -> Optional[HealthcareFacility]:
        """Parse Google Places API response into HealthcareFacility object"""
        
        try:
            # Extract basic information
            name = place.get('name', 'Unknown')
            place_id = place.get('place_id', '')
            
            # Filter out irrelevant results (hotels, restaurants, etc.)
            if self._should_exclude_facility(place, name):
                return None
            
            # Location information
            location = place.get('geometry', {}).get('location', {})
            lat = location.get('lat', 0.0)
            lng = location.get('lng', 0.0)
            
            # Address
            address = place.get('vicinity', place.get('formatted_address', ''))
            
            # Rating information
            rating = place.get('rating')
            rating_count = place.get('user_ratings_total')
            
            # Business status
            business_status = place.get('business_status', 'OPERATIONAL')
            
            # Create maps URL
            maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
            
            # Phone number (if available)
            phone = place.get('formatted_phone_number')
            
            return HealthcareFacility(
                name=name,
                address=address,
                phone=phone,
                rating=rating,
                rating_count=rating_count,
                place_id=place_id,
                latitude=lat,
                longitude=lng,
                facility_type=facility_type,
                business_status=business_status,
                maps_url=maps_url
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse place data: {e}")
            return None
    
    def _should_exclude_facility(self, place: Dict, name: str) -> bool:
        """Check if facility should be excluded (hotels, restaurants, etc.)"""
        
        # Keywords that indicate non-healthcare facilities
        exclude_keywords = [
            'hotel', 'inn', 'resort', 'restaurant', 'cafe', 'bar', 'pub',
            'store', 'shop', 'mall', 'gym', 'fitness', 'spa', 'salon',
            'bank', 'atm', 'gas station', 'school', 'university'
        ]
        
        name_lower = name.lower()
        
        # Check name for exclude keywords
        if any(keyword in name_lower for keyword in exclude_keywords):
            return True
        
        # Check place types
        place_types = place.get('types', [])
        exclude_types = [
            'lodging', 'restaurant', 'food', 'store', 'shopping_mall',
            'gas_station', 'bank', 'atm', 'school', 'university',
            'gym', 'spa', 'beauty_salon'
        ]
        
        if any(exclude_type in place_types for exclude_type in exclude_types):
            return True
        
        # Additional filtering for healthcare relevance
        healthcare_keywords = [
            'hospital', 'medical', 'clinic', 'doctor', 'physician', 'health',
            'emergency', 'urgent care', 'rehabilitation', 'therapy', 'center'
        ]
        
        # If name doesn't contain any healthcare keywords, be suspicious
        if not any(keyword in name_lower for keyword in healthcare_keywords):
            # Check if it has healthcare-related types
            healthcare_types = ['hospital', 'doctor', 'health', 'physiotherapist']
            if not any(hc_type in place_types for hc_type in healthcare_types):
                return True
        
        return False
    
    def _deduplicate_facilities(self, facilities: List[HealthcareFacility]) -> List[HealthcareFacility]:
        """Remove duplicate facilities based on name and location"""
        
        seen = set()
        unique_facilities = []
        
        for facility in facilities:
            # Create a unique key based on name and approximate location
            key = (
                facility.name.lower().strip(),
                round(facility.latitude, 4),
                round(facility.longitude, 4)
            )
            
            if key not in seen:
                seen.add(key)
                unique_facilities.append(facility)
        
        return unique_facilities
    
    def _sort_facilities(self, facilities: List[HealthcareFacility], location_coords: Dict) -> List[HealthcareFacility]:
        """Sort facilities by review count first, then rating and distance"""
        
        def sort_key(facility):
            # Prioritize review count (more reviews = more reliable)
            review_count = facility.rating_count or 0
            
            # Rating score (prefer higher ratings)
            rating_score = facility.rating or 0
            
            # Distance calculation (closer = better)
            if location_coords:
                lat_diff = facility.latitude - location_coords['lat']
                lng_diff = facility.longitude - location_coords['lng']
                distance_score = -(lat_diff**2 + lng_diff**2)  # Negative for closer = better
            else:
                distance_score = 0
            
            # Combined score: review count is primary, then rating, then distance
            # Normalize review count to 0-5 scale for fair comparison
            normalized_reviews = min(review_count / 100, 5.0)  # Cap at 5 for 500+ reviews
            
            return (normalized_reviews * 0.5) + (rating_score * 0.3) + (distance_score * 0.2)
        
        return sorted(facilities, key=sort_key, reverse=True)
    
    def _add_distance_info(self, facilities: List[HealthcareFacility], location_coords: Dict) -> List[HealthcareFacility]:
        """Add distance information to facilities"""
        
        if not location_coords:
            return facilities
        
        for facility in facilities:
            # Simple distance calculation (Haversine formula approximation)
            lat1, lng1 = location_coords['lat'], location_coords['lng']
            lat2, lng2 = facility.latitude, facility.longitude
            
            # Approximate distance in miles
            lat_diff = lat2 - lat1
            lng_diff = lng2 - lng1
            distance_approx = ((lat_diff**2 + lng_diff**2)**0.5) * 69  # Rough miles conversion
            
            facility.distance_miles = round(distance_approx, 1)
        
        return facilities

class HealthcareLocatorAgent:
    """Main Healthcare Locator Agent"""
    
    def __init__(self, gemini_api_key: str = None, google_places_api_key: str = None):
        """Initialize the Healthcare Locator Agent"""
        
        load_dotenv()
        
        # API Configuration
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.google_places_api_key = google_places_api_key or os.getenv("GOOGLE_PLACES_API_KEY")
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        if not self.google_places_api_key:
            raise ValueError("Google Places API key is required")
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ Healthcare Locator Agent initialized")
    
    def _initialize_components(self):
        """Initialize LLM and API components"""
        try:
            # Initialize Gemini
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize components
            self.location_parser = IntelligentLocationParser(self.gemini_model)
            self.places_connector = GooglePlacesConnector(self.google_places_api_key)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def find_healthcare_facilities(self, query: str) -> Dict[str, Any]:
        """Main method to find healthcare facilities based on natural language query"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Parse the location query using LLM
            logger.info(f"🧠 Parsing query: {query[:50]}...")
            location_query = await self.location_parser.parse_location_query(query)
            
            logger.info(f"🎯 Parsed - Facility: {[f.value for f in location_query.facility_types]}, Location: {location_query.location}")
            
            # Step 2: Search for facilities using Google Places API
            logger.info(f"🔍 Searching for facilities...")
            facilities, error = await self.places_connector.search_facilities(location_query)
            
            if error:
                return self._create_error_response(error, start_time)
            
            if not facilities:
                return self._create_no_results_response(location_query, start_time)
            
            # Step 3: Format response
            response = self._format_facility_response(facilities, location_query, start_time)
            
            logger.info(f"✅ Found {len(facilities)} facilities in {response['processing_time']:.2f}s")
            
            return response
            
        except Exception as e:
            logger.exception(f"❌ Error finding facilities: {e}")
            return self._create_error_response(f"Search failed: {str(e)}", start_time)
    
    def _format_facility_response(
        self, 
        facilities: List[HealthcareFacility], 
        location_query: LocationQuery, 
        start_time: datetime
    ) -> Dict[str, Any]:
        """Format the facility search response"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Group facilities by type for better presentation
        grouped_facilities = {}
        for facility in facilities:
            facility_type = facility.facility_type
            if facility_type not in grouped_facilities:
                grouped_facilities[facility_type] = []
            grouped_facilities[facility_type].append(facility)
        
        # Create formatted response text
        response_text = self._generate_response_text(facilities, location_query)
        
        # Create structured facility data
        facility_data = []
        for facility in facilities:
            facility_info = {
                "name": facility.name,
                "address": facility.address,
                "phone": facility.phone,
                "rating": facility.rating,
                "rating_count": facility.rating_count,
                "facility_type": facility.facility_type,
                "maps_url": facility.maps_url,
                "distance_miles": facility.distance_miles,
                "business_status": facility.business_status
            }
            facility_data.append(facility_info)
        
        return {
            "success": True,
            "query": location_query.original_query,
            "parsed_location": location_query.location,
            "facility_types_searched": [f.value for f in location_query.facility_types],
            "total_results": len(facilities),
            "urgency_level": location_query.urgency,
            "search_radius_miles": location_query.radius_miles,
            "response": response_text,
            "facilities": facility_data,
            "grouped_facilities": {k: len(v) for k, v in grouped_facilities.items()},
            "processing_time": processing_time,
            "confidence": location_query.confidence
        }
    
    def _generate_response_text(self, facilities: List[HealthcareFacility], location_query: LocationQuery) -> str:
        """Generate human-readable response text"""
        
        if not facilities:
            return f"I couldn't find any {', '.join([f.value for f in location_query.facility_types])} near {location_query.location}. You may want to try expanding your search radius or searching in nearby areas."
        
        # Create response header
        facility_types_str = ', '.join([f.value.replace('_', ' ').title() for f in location_query.facility_types])
        location_str = location_query.location if location_query.location else "your area"
        
        response_parts = [
            f"I found {len(facilities)} {facility_types_str.lower()} facilities near {location_str}:",
            ""
        ]
        
        # Add top facilities with details
        for i, facility in enumerate(facilities[:8], 1):  # Show top 8
            facility_info = [f"**{i}. {facility.name}**"]
            
            if facility.address:
                facility_info.append(f"   📍 {facility.address}")
            
            if facility.phone:
                facility_info.append(f"   📞 {facility.phone}")
            
            # Show reviews first (primary sorting criteria), then rating
            if facility.rating_count and facility.rating:
                stars = "⭐" * int(facility.rating)
                facility_info.append(f"   👥 {facility.rating_count} reviews | {stars} {facility.rating}/5")
            elif facility.rating_count:
                facility_info.append(f"   👥 {facility.rating_count} reviews")
            elif facility.rating:
                stars = "⭐" * int(facility.rating)
                facility_info.append(f"   {stars} {facility.rating}/5")
            
            if facility.distance_miles:
                facility_info.append(f"   📏 {facility.distance_miles} miles away")
            
            facility_info.append(f"   🗺️ [View on Maps]({facility.maps_url})")
            facility_info.append("")
            
            response_parts.extend(facility_info)
        
        # Add additional info if more results available
        if len(facilities) > 8:
            response_parts.append(f"*...and {len(facilities) - 8} more facilities found*")
        
        # Add urgency-specific advice
        if location_query.urgency == "emergency":
            response_parts.extend([
                "",
                "🚨 **For medical emergencies, call 911 immediately or go to the nearest emergency room.**"
            ])
        elif location_query.urgency == "urgent":
            response_parts.extend([
                "",
                "⚡ **For urgent care needs, I recommend calling ahead to check availability and wait times.**"
            ])
        
        return "\n".join(response_parts)
    
    def _create_error_response(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Create error response"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": False,
            "error": error_message,
            "response": f"I encountered an issue while searching for healthcare facilities: {error_message}",
            "facilities": [],
            "processing_time": processing_time
        }
    
    def _create_no_results_response(self, location_query: LocationQuery, start_time: datetime) -> Dict[str, Any]:
        """Create response when no facilities are found"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        facility_types_str = ', '.join([f.value.replace('_', ' ') for f in location_query.facility_types])
        
        response_text = f"""I couldn't find any {facility_types_str} facilities near {location_query.location or 'your location'} within {location_query.radius_miles} miles.

**Suggestions:**
• Try searching with a larger radius
• Check nearby cities or ZIP codes  
• Search for more general facility types (e.g., "hospitals" instead of "neurotherapy clinics")
• Contact your insurance provider for in-network options
• Use online directories like Healthgrades or your insurance website"""
        
        return {
            "success": True,
            "query": location_query.original_query,
            "total_results": 0,
            "response": response_text,
            "facilities": [],
            "processing_time": processing_time,
            "suggestions": [
                "Expand search radius",
                "Try nearby locations",
                "Search for general facility types",
                "Contact insurance provider"
            ]
        }


# Enhanced Testing Interface
async def test_locator_interface():
    """Interactive testing interface for Healthcare Locator Agent"""
    
    print("=" * 80)
    print("🏥 HEALTHCARE LOCATOR AGENT - TESTING INTERFACE")
    print("=" * 80)
    
    # Check for API keys
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
    print("🚀 Initializing Healthcare Locator Agent...")
    try:
        agent = HealthcareLocatorAgent(gemini_key, places_key)
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    print("\n🎯 **FEATURES:**")
    print("   • **Smart Query Parsing** - Understands natural language location requests")
    print("   • **Google Places Integration** - Real-time facility data with ratings & reviews")
    print("   • **Multi-Facility Search** - Hospitals, rehab centers, clinics, urgent care")
    print("   • **Distance & Rating Sorting** - Best facilities first")
    print("   • **Maps Integration** - Direct links to Google Maps")
    
    print("\n📋 **EXAMPLE QUERIES:**")
    print("   • 'hospitals near 02134'")
    print("   • 'TBI rehabilitation centers in Boston MA'")
    print("   • 'urgent care clinics near me in Los Angeles'")
    print("   • 'neurologists in New York City'")
    print("   • 'physical therapy near 90210'")
    
    print("\n💡 **COMMANDS:**")
    print("   • Type your location query naturally")
    print("   • 'test' - Run sample queries")
    print("   • 'quit' - Exit")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\n🔍 Enter your healthcare facility search: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n👋 Thank you for testing the Healthcare Locator Agent!")
                break
            
            elif user_input.lower() == 'test':
                # Run sample queries
                test_queries = [
                    "hospitals near 02134",
                    "TBI rehabilitation centers in Boston",
                    "urgent care near 90210"
                ]
                
                for query in test_queries:
                    print(f"\n🧪 **Testing:** '{query}'")
                    result = await agent.find_healthcare_facilities(query)
                    print(f"**Result:** {result['success']} - {len(result.get('facilities', []))} facilities found")
                continue
            
            # Process the query
            print(f"\n🤖 Processing: '{user_input}'...")
            start_time = datetime.now()
            
            result = await agent.find_healthcare_facilities(user_input)
            
            if result["success"]:
                print(f"\n🏥 **Results:**")
                print(result['response'])
                
                print(f"\n📊 **Search Details:**")
                print(f"   📍 Location: {result.get('parsed_location', 'N/A')}")
                print(f"   🏢 Facility Types: {', '.join(result.get('facility_types_searched', []))}")
                print(f"   📈 Total Results: {result['total_results']}")
                print(f"   ⏱️ Processing Time: {result['processing_time']:.2f}s")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.2f}")
                
                if result.get('grouped_facilities'):
                    print(f"   📋 By Type: {result['grouped_facilities']}")
                
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
    
    asyncio.run(test_locator_interface())