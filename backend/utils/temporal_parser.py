"""Temporal query parser for extracting date constraints from user queries."""
import re
from datetime import datetime
from typing import Optional, Dict, Tuple
from dateutil.relativedelta import relativedelta

class TemporalQueryParser:
    """Extract date ranges and temporal constraints from natural language queries."""
    
    def __init__(self):
        """Initialize temporal patterns."""
        self.current_year = datetime.now().year
        
        # Decade patterns
        self.decade_pattern = re.compile(r"(\d{2})'?s|(\d{4})s")
        
        # Year patterns
        self.year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
        self.year_range_pattern = re.compile(r"(\d{4})\s*[-–—to]\s*(\d{4})")
        
        # Relative time patterns
        self.relative_patterns = {
            r"last\s+year": (-1, 0),
            r"this\s+year": (0, 0),
            r"past\s+(\d+)\s+years?": None,  # Dynamic
            r"recent(?:ly)?": (-2, 0),
        }
        
        # Named periods
        self.named_periods = {
            "classic": (1950, 1989),
            "golden age": (1930, 1959),
            "modern": (2000, self.current_year),
            "contemporary": (2015, self.current_year),
            "vintage": (1920, 1969),
        }
    
    def parse(self, query: str) -> Dict[str, Optional[int]]:
        """
        Parse query and extract temporal constraints.
        
        Returns:
            Dict with 'start_year' and 'end_year' keys
        """
        query_lower = query.lower()
        
        # Check for explicit year ranges
        year_range = self._extract_year_range(query)
        if year_range:
            return {"start_year": year_range[0], "end_year": year_range[1]}
        
        # Check for single years
        single_year = self._extract_single_year(query)
        if single_year:
            return {"start_year": single_year, "end_year": single_year}
        
        # Check for decades
        decade = self._extract_decade(query)
        if decade:
            return {"start_year": decade[0], "end_year": decade[1]}
        
        # Check for relative time
        relative = self._extract_relative_time(query_lower)
        if relative:
            return {"start_year": relative[0], "end_year": relative[1]}
        
        # Check for named periods
        named = self._extract_named_period(query_lower)
        if named:
            return {"start_year": named[0], "end_year": named[1]}
        
        return {"start_year": None, "end_year": None}
    
    def _extract_year_range(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract year ranges like '2015-2020' or '1990 to 1999'."""
        match = self.year_range_pattern.search(query)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            return (min(start, end), max(start, end))
        return None
    
    def _extract_single_year(self, query: str) -> Optional[int]:
        """Extract single year mentions."""
        # Avoid matching decades (e.g., "1990s")
        if re.search(r"\d{4}s", query):
            return None
        
        match = self.year_pattern.search(query)
        return int(match.group(1)) if match else None
    
    def _extract_decade(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract decades like '80s', '1990s', '2000s'."""
        match = self.decade_pattern.search(query)
        if match:
            if match.group(1):  # "80s" format
                decade = int(match.group(1))
                # Determine century
                if decade >= 20 and decade <= 29:
                    start = 2000 + decade
                else:
                    start = 1900 + decade
            else:  # "1990s" format
                start = int(match.group(2))
            
            return (start, start + 9)
        return None
    
    def _extract_relative_time(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract relative time expressions."""
        for pattern, offset in self.relative_patterns.items():
            match = re.search(pattern, query)
            if match:
                if offset is None:  # Dynamic pattern
                    num_years = int(match.group(1))
                    return (self.current_year - num_years, self.current_year)
                else:
                    start_offset, end_offset = offset
                    return (
                        self.current_year + start_offset,
                        self.current_year + end_offset
                    )
        return None
    
    def _extract_named_period(self, query: str) -> Optional[Tuple[int, int]]:
        """Extract named time periods."""
        for period_name, (start, end) in self.named_periods.items():
            if period_name in query:
                return (start, end)
        return None
    
    def format_for_tmdb(self, temporal_constraints: Dict[str, Optional[int]]) -> Dict[str, str]:
        """
        Format temporal constraints for TMDb API.
        
        Returns:
            Dict with 'primary_release_date.gte' and 'primary_release_date.lte'
        """
        result = {}
        
        if temporal_constraints.get("start_year"):
            result["primary_release_date.gte"] = f"{temporal_constraints['start_year']}-01-01"
        
        if temporal_constraints.get("end_year"):
            result["primary_release_date.lte"] = f"{temporal_constraints['end_year']}-12-31"
        
        return result

# Global parser instance
temporal_parser = TemporalQueryParser()
