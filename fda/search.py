"""Search agent: web search (DuckDuckGo, no key) and weather (Open-Meteo, no key), answered in a spoken sentence."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from . import brain

log = logging.getLogger("fda.search")

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
       81: "rain showers", 82: "heavy rain showers", 95: "thunderstorms", 96: "thunderstorms with hail",
       99: "thunderstorms with hail"}


class SearchAgent:
    def __init__(self, cfg, servers):
        self.cfg, self.servers = cfg, servers
        self._geo: dict[str, tuple] = {}

    async def web(self, query: str) -> tuple[AsyncIterator[str] | None, list[dict], str]:
        """-> (stream of the spoken answer, sources for the display, fallback text if no stream)."""
        def run():
            from ddgs import DDGS
            return DDGS().text(query, max_results=5)
        try:
            results = await asyncio.wait_for(asyncio.to_thread(run), timeout=8)
        except Exception as e:
            log.warning("search failed: %s", e)
            return None, [], "Sorry, I couldn't reach the internet to look that up."
        if not results:
            return None, [], "I couldn't find anything on that."
        sources = [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:240]}
                   for r in results]
        text = "\n".join(f"- {s['title']}: {s['snippet']}" for s in sources)
        prompt = brain.SEARCH_PROMPT.format(name=self.cfg.name, today=time.strftime("%B %d, %Y"), q=query,
                                            results=text)
        stream = self.servers.chat_stream([{"role": "user", "content": prompt}], max_tokens=90, temperature=0.2)
        return stream, sources, ""

    async def weather(self, place: str | None) -> str:
        place = (place or self.cfg.location).strip()
        http = self.servers.web
        try:
            if place not in self._geo:
                r = await http.get("https://geocoding-api.open-meteo.com/v1/search",
                                   params={"name": place, "count": 1}, timeout=6)
                g = (r.json().get("results") or [None])[0]
                if not g:
                    return f"I couldn't find a place called {place}."
                self._geo[place] = (g["latitude"], g["longitude"], g["name"])
            lat, lon, name = self._geo[place]
            r = await http.get("https://api.open-meteo.com/v1/forecast", timeout=6, params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "current": "temperature_2m,weather_code,apparent_temperature",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max", "forecast_days": 1})
            w = r.json()
            c, d = w["current"], w["daily"]
            desc = WMO.get(int(c["weather_code"]), "")
            rain = d["precipitation_probability_max"][0]
            return (f"In {name} it's {round(c['temperature_2m'])} degrees and {desc}, feels like "
                    f"{round(c['apparent_temperature'])}. Today's high is {round(d['temperature_2m_max'][0])}"
                    f"{f', with a {rain} percent chance of rain' if rain is not None else ''}.")
        except Exception as e:
            log.warning("weather failed: %s", e)
            return "Sorry, I couldn't get the weather right now."
