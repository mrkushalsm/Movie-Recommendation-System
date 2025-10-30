"""Wikipedia client for extracting structured movie information."""
import wikipedia
import re
from typing import Dict, List, Optional
from utils.cache_manager import cache_manager

class WikipediaClient:
    """Extract structured information from Wikipedia movie pages."""
    
    def __init__(self):
        """Initialize Wikipedia client."""
        wikipedia.set_lang("en")
    
    @cache_manager.cached("wiki_search", ttl=86400)
    def search_movie(self, movie_title: str, year: Optional[int] = None) -> Optional[str]:
        """
        Search for a movie page on Wikipedia.
        
        Returns:
            Page title if found, None otherwise
        """
        try:
            search_query = f"{movie_title} {year} film" if year else f"{movie_title} film"
            results = wikipedia.search(search_query, results=5)
            
            # Try to find the most relevant result
            for result in results:
                if "film" in result.lower() and movie_title.lower() in result.lower():
                    return result
            
            return results[0] if results else None
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None
    
    @cache_manager.cached("wiki_page", ttl=86400)
    def get_page_content(self, page_title: str) -> Optional[str]:
        """Get full Wikipedia page content."""
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            return page.content
        except Exception as e:
            print(f"Wikipedia page error: {e}")
            return None
    
    def extract_sections(self, page_title: str) -> Dict[str, str]:
        """
        Extract structured sections from Wikipedia page.
        
        Returns:
            Dict with keys: plot, themes, production, reception, cast
        """
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            content = page.content
            
            sections = {
                "plot": self._extract_section(content, ["Plot", "Synopsis", "Story"]),
                "themes": self._extract_section(content, ["Themes", "Analysis", "Interpretation"]),
                "production": self._extract_section(content, ["Production", "Development", "Filming"]),
                "reception": self._extract_section(content, ["Reception", "Critical response", "Box office"]),
                "cast": self._extract_section(content, ["Cast", "Starring"]),
            }
            
            # Add summary as fallback
            sections["summary"] = page.summary
            
            return {k: v for k, v in sections.items() if v}
        
        except Exception as e:
            print(f"Section extraction error: {e}")
            return {}
    
    def _extract_section(self, content: str, possible_headers: List[str]) -> Optional[str]:
        """Extract a specific section from Wikipedia content."""
        for header in possible_headers:
            # Match section headers (== Header ==)
            pattern = rf"==+\s*{header}\s*==+(.*?)(?===+|$)"
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            
            if match:
                section_text = match.group(1).strip()
                # Remove subsection markers
                section_text = re.sub(r"===.*?===", "", section_text)
                # Clean up
                section_text = re.sub(r"\n+", "\n", section_text)
                return section_text[:2000]  # Limit length
        
        return None
    
    def get_movie_info(self, movie_title: str, year: Optional[int] = None) -> Dict[str, str]:
        """
        Get comprehensive movie information from Wikipedia.
        
        Returns:
            Dict with plot, themes, production info, etc.
        """
        page_title = self.search_movie(movie_title, year)
        
        if not page_title:
            return {}
        
        return self.extract_sections(page_title)
    
    @cache_manager.cached("wiki_summary", ttl=86400)
    def get_summary(self, movie_title: str, year: Optional[int] = None) -> Optional[str]:
        """Get movie summary from Wikipedia."""
        page_title = self.search_movie(movie_title, year)
        
        if not page_title:
            return None
        
        try:
            page = wikipedia.page(page_title, auto_suggest=False)
            return page.summary
        except Exception as e:
            print(f"Wikipedia summary error: {e}")
            return None

# Global client instance
wiki_client = WikipediaClient()
