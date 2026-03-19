from memos.configs.chunker import SentenceChunkerConfig
from memos.dependency import require_python_package
from memos.log import get_logger

from .base import BaseChunker, Chunk


logger = get_logger(__name__)


class SentenceChunker(BaseChunker):
    """Sentence-based text chunker."""

    def _simple_chunk(self, text: str) -> list[Chunk]:
        """Fallback splitter when chonkie or tokenizer initialization is unavailable."""
        protected_text, url_map = self.protect_urls(text)
        if not protected_text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        text_len = len(protected_text)

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            if end < text_len:
                for separator in ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "]:
                    last_sep = protected_text.rfind(separator, start, end)
                    if last_sep != -1:
                        end = last_sep + len(separator)
                        break

            chunk_text = protected_text[start:end].strip()
            if chunk_text:
                restored_text = self.restore_urls(chunk_text, url_map)
                chunks.append(
                    Chunk(
                        text=restored_text,
                        token_count=len(restored_text),
                        sentences=[restored_text],
                    )
                )

            start = max(start + 1, end - self.config.chunk_overlap)

        return chunks

    @require_python_package(
        import_name="chonkie",
        install_command="pip install chonkie",
        install_link="https://docs.chonkie.ai/python-sdk/getting-started/installation",
    )
    def __init__(self, config: SentenceChunkerConfig):
        from chonkie import SentenceChunker as ChonkieSentenceChunker

        self.config = config
        self.chunker = None

        # Try new API first (v1.4.0+)
        try:
            self.chunker = ChonkieSentenceChunker(
                tokenizer=config.tokenizer_or_token_counter,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_sentences_per_chunk=config.min_sentences_per_chunk,
            )
        except (TypeError, AttributeError) as e:
            # Fallback to old API (<v1.4.0)
            logger.debug(f"Falling back to old chonkie API: {e}")
            try:
                self.chunker = ChonkieSentenceChunker(
                    tokenizer_or_token_counter=config.tokenizer_or_token_counter,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    min_sentences_per_chunk=config.min_sentences_per_chunk,
                )
            except Exception as fallback_error:
                logger.warning(
                    "Failed to initialize chonkie sentence chunker; using simple fallback: %s",
                    fallback_error,
                )
        except Exception as e:
            logger.warning(
                "Failed to initialize chonkie sentence chunker; using simple fallback: %s",
                e,
            )

        logger.info(f"Initialized SentenceChunker with config: {config}")

    def chunk(self, text: str) -> list[str] | list[Chunk]:
        """Chunk the given text into smaller chunks based on sentences."""
        if self.chunker is None:
            return self._simple_chunk(text)

        protected_text, url_map = self.protect_urls(text)
        chonkie_chunks = self.chunker.chunk(protected_text)

        chunks = []
        for c in chonkie_chunks:
            chunk = Chunk(text=c.text, token_count=c.token_count, sentences=c.sentences)
            chunk = self.restore_urls(chunk.text, url_map)
            chunks.append(chunk)

        logger.debug(f"Generated {len(chunks)} chunks from input text")
        return chunks
