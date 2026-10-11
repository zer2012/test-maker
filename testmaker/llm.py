"""Client LLM asynchrone pour le teacher (Ollama, OpenRouter...) et, plus tard, le student servi par llama-server."""

import asyncio
import os

from openai import AsyncOpenAI


class LLM:
    """Client minimal pour tout endpoint compatible OpenAI (teacher ou student servi localement)."""

    def __init__(self, model: str, base_url: str, api_key_env: str, concurrency: int = 8,
                 timeout: float = 300):
        self.model = model
        # La clé est lue dans une variable d'environnement (jamais écrite dans la config).
        # Ollama n'en a pas besoin mais le SDK exige une valeur : d'où "sk-no-key".
        # max_retries : le SDK réessaie seul en cas d'erreur réseau, de 429 ou de 5xx.
        self.client = AsyncOpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "sk-no-key"),
                                  max_retries=6, timeout=timeout)
        # Limite le nombre de requêtes simultanées (1 pour Ollama sur CPU, 16 pour une API).
        self.sem = asyncio.Semaphore(concurrency)

    async def complete(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        """Une requête de chat ; renvoie le texte de la réponse."""
        async with self.sem:  # attend qu'une « place » se libère avant d'envoyer
            resp = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
        return resp.choices[0].message.content or ""

    async def sample(self, messages: list[dict], n: int, temperature: float, max_tokens: int) -> list[str]:
        """Demande n réponses indépendantes au même prompt (candidats du rejection sampling)."""
        # n requêtes séparées plutôt que le paramètre `n`, que beaucoup de fournisseurs ignorent.
        results = await asyncio.gather(
            *(self.complete(messages, temperature, max_tokens) for _ in range(n)),
            return_exceptions=True,  # une requête en échec n'annule pas les autres
        )
        return [r for r in results if isinstance(r, str)]  # on ne garde que les succès


def teacher_llm(cfg: dict, provider: str | None = None) -> LLM:
    """Construit le client du teacher à partir du profil choisi (config ou option --provider)."""
    t = cfg["teacher"]
    p = t["providers"][provider or t["provider"]]
    return LLM(p["model"], p["base_url"], p["api_key_env"], p.get("concurrency", 8), p.get("timeout_s", 300))
