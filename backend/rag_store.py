import os
import math
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Persistent database directory
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "argo_ocean_knowledge"

# Core Oceanographic Knowledge Corpus
OCEAN_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "argo_cycle_mechanics",
        "title": "ARGO Float Lifecycle and Profiling Mechanism",
        "category": "instrumentation",
        "text": """An ARGO float follows a continuous 9-10 day autonomous operational cycle. 
1. Deployment: The float is placed in the ocean and sinks to a parking depth (typically 1,000 meters / 1000 dbar).
2. Neutral Drift: It drifts with deep ocean currents for approximately 8 to 9 days at 1,000m.
3. Descent to Profile Depth: On day 9, it pumps oil into an internal reservoir to sink down to 2,000 meters (2000 dbar).
4. Ascent & Profiling: It pumps oil into an external bladder, becoming buoyant, and ascends over 6 hours while continuously recording Conductivity (salinity), Temperature, and Pressure (depth) via its CTD sensor.
5. Satellite Transmission: At the surface, the float uses GPS to fix its position and transmits all recorded profile data back to shore via Iridium or Argos satellite telemetry within 15-30 minutes, then repeats the cycle."""
    },
    {
        "id": "ocean_temperature_thermocline",
        "title": "Ocean Thermal Structure and Thermocline",
        "category": "physics",
        "text": """The ocean's vertical temperature profile consists of three primary layers:
1. Mixed Surface Layer (0-100m): Homogeneous temperature caused by wind turbulence, waves, and solar radiation. Temperatures typically range from -1.8°C (polar) to 30°C (equatorial).
2. Thermocline (100m-1,000m): A zone of rapid temperature decrease with depth. In tropical and subtropical oceans, the thermocline separates warm surface waters from icy deep waters. Below the seasonal thermocline lies the permanent thermocline.
3. Deep Ocean Layer (1,000m-4,000m+): Cold, dense, stable water where temperatures globally hover between 0°C and 3°C, regardless of surface latitude.
ARGO floats measure this vertical thermal structure to calculate global Ocean Heat Content (OHC) and monitor marine heatwaves."""
    },
    {
        "id": "ocean_salinity_halocline",
        "title": "Ocean Salinity and Halocline Dynamics",
        "category": "physics",
        "text": """Ocean salinity is measured using the Practical Salinity Scale (PSU, roughly parts per thousand). The global open-ocean average is approximately 35 PSU (ranging from 33 to 37 PSU).
1. Halocline: A vertical layer in the water column where salinity changes rapidly with depth. In high-latitude polar waters, salinity increases with depth as surface water is freshened by melting ice and precipitation. In subtropical gyres (Atlantic, Pacific), evaporation exceeds precipitation, creating a high-salinity surface layer above fresher deeper water.
2. Density (Pycnocline): Density is a combined function of temperature and salinity (sigma-t). Cold, salty water is the densest and sinks, driving global thermohaline circulation (the global ocean conveyor belt)."""
    },
    {
        "id": "major_ocean_currents",
        "title": "Global Ocean Circulation and Major Currents",
        "category": "circulation",
        "text": """ARGO floats track intermediate and deep ocean currents worldwide:
1. Antarctic Circumpolar Current (ACC): The strongest and largest ocean current on Earth, flowing uninterrupted eastward around Antarctica, connecting the Atlantic, Indian, and Pacific basins.
2. Gulf Stream & North Atlantic Drift: Part of the Atlantic Meridional Overturning Circulation (AMOC), transporting warm equatorial water northward, influencing European climate.
3. Kuroshio Current: The western boundary current of the North Pacific, transporting warm tropical waters northward toward Japan.
4. Indian Ocean Monsoon Reversals: Surface currents in the northern Indian Ocean reverse direction biannually with the summer and winter monsoons, while the southern Indian Ocean features the Agulhas Current."""
    },
    {
        "id": "bgc_argo_biogeochemical",
        "title": "Biogeochemical Argo (BGC-Argo)",
        "category": "biogeochemistry",
        "text": """BGC-Argo floats are equipped with additional bio-optical and chemical sensors to observe ocean health and carbon uptake:
1. Dissolved Oxygen (DO): Measured with optical optodes to monitor ocean deoxygenation and oxygen minimum zones (OMZs).
2. Nitrate (NO3-): Optical spectrophotometers detect nutrient availability for marine primary productivity.
3. pH: ISFET sensors measure ocean acidification caused by anthropogenic CO2 absorption.
4. Chlorophyll-a: Fluorometers quantify phytoplankton biomass in the euphotic zone.
5. Suspended Particles: Optical backscattering sensors monitor particulate organic carbon export into the deep ocean (biological carbon pump)."""
    },
    {
        "id": "argo_data_system_erddap",
        "title": "ARGO Data Ingestion via NOAA ERDDAP",
        "category": "data_systems",
        "text": """ARGO float data is collected and distributed globally through Global Data Assembly Centers (GDACs: Coriolis in France and US GODAE in Monterey).
NOAA ERDDAP provides access to ARGO observations via the tabledap protocol:
- Server URL: https://polarwatch.noaa.gov/erddap/tabledap/argoFloats.json
- Standard query variables: time (ISO-8601), latitude, longitude, depth (or pres in dbar), temperature (temp in degC), salinity (psal in PSU), platform_number (WMO ID).
- Quality Control (QC): Real-time quality control flags 1 (good data), 2 (probably good), 3 (probably bad), 4 (bad data)."""
    }
]


class ArgoRAGStore:
    """
    ChromaDB-powered Retrieval-Augmented Generation (RAG) Store
    for ARGO-AI oceanographic domain knowledge.
    """
    def __init__(self, persist_dir: str = CHROMA_DATA_PATH):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self._ensure_knowledge_seeded()

    def _ensure_knowledge_seeded(self):
        """Seed the knowledge base if empty."""
        try:
            count = self.collection.count()
            if count == 0:
                print(f"[RAG] Seeding ChromaDB collection '{COLLECTION_NAME}' with initial ocean knowledge...")
                docs = [item["text"] for item in OCEAN_KNOWLEDGE_DOCUMENTS]
                ids = [item["id"] for item in OCEAN_KNOWLEDGE_DOCUMENTS]
                metadatas = [{"title": item["title"], "category": item["category"]} for item in OCEAN_KNOWLEDGE_DOCUMENTS]
                self.collection.add(
                    documents=docs,
                    ids=ids,
                    metadatas=metadatas
                )
                print(f"[RAG] Successfully seeded {len(ids)} documents into ChromaDB.")
            else:
                print(f"[RAG] ChromaDB collection '{COLLECTION_NAME}' already contains {count} documents.")
        except Exception as e:
            print(f"[RAG] Note during ChromaDB seeding: {e}")

    def query(self, query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for relevant oceanographic context.
        Falls back gracefully to keyword matching if vector query is initializing.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count() or 1)
            )
            
            extracted = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                distances = results.get("distances", [[]])[0] if results.get("distances") else []
                
                for i, doc in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    dist = distances[i] if i < len(distances) else 0.0
                    extracted.append({
                        "content": doc,
                        "metadata": meta,
                        "score": dist
                    })
                return extracted
        except Exception as e:
            print(f"[RAG] Vector query notice: {e}, falling back to semantic keyword retrieval")
            
        # Robust fallback: lexical relevance search over the documents
        query_words = set(query_text.lower().split())
        scored_docs = []
        for item in OCEAN_KNOWLEDGE_DOCUMENTS:
            text_lower = item["text"].lower()
            score = sum(1 for w in query_words if len(w) > 3 and w in text_lower)
            if score > 0:
                scored_docs.append((score, item))
                
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top = scored_docs[:n_results] or [(1, OCEAN_KNOWLEDGE_DOCUMENTS[0])]
        return [{
            "content": item["text"],
            "metadata": {"title": item["title"], "category": item["category"]},
            "score": score
        } for score, item in top]

    def format_context_for_prompt(self, query_text: str, n_results: int = 2) -> str:
        """
        Retrieve and format top RAG documents into a clean context string for the LLM.
        """
        hits = self.query(query_text, n_results=n_results)
        if not hits:
            return ""
        
        context_parts = []
        for i, hit in enumerate(hits, 1):
            title = hit.get("metadata", {}).get("title", f"Reference {i}")
            content = hit["content"].strip()
            context_parts.append(f"--- [Domain Knowledge: {title}] ---\n{content}")
            
        return "\n\n".join(context_parts)

    def count(self) -> int:
        """Returns total document count in ChromaDB collection."""
        try:
            return self.collection.count()
        except Exception:
            return len(OCEAN_KNOWLEDGE_DOCUMENTS)


# Singleton instance
rag_store = ArgoRAGStore()
