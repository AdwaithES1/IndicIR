import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# File paths
input_file = r"D:\indicIR\Datasets\SciFact\data\data\corpus.jsonl"
output_file = r"D:\indicIR\Datasets\SciFact\data\data\corpus_translated.jsonl"

# Model and language codes
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "mal_Mlym"

# Batch size
BATCH_SIZE = 2


# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    attn_implementation="flash_attention_2" if DEVICE == "cuda" else None
).to(DEVICE)
ip = IndicProcessor(inference=True)

def translate_abstracts(texts):
    batch = ip.preprocess_batch(texts, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=512,
            num_beams=5,
            num_return_sequences=1
        )
    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
    return translations

def main():
    # Count total lines for tqdm progress bar
    with open(input_file, "r", encoding="utf-8") as fin:
        total_lines = sum(1 for _ in fin)

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        buffer = []
        entries_buffer = []
        processed = 0

        for line in tqdm(fin, total=total_lines, desc="Translating"):
            entry = json.loads(line)
            abstract_text = entry.get("abstract", None)
            # Handle both list and string formats
            if isinstance(abstract_text, list):
                abstract_text = " ".join(abstract_text)
            elif not isinstance(abstract_text, str):
                abstract_text = ""
            if abstract_text.strip():
                buffer.append(abstract_text)
                entries_buffer.append(entry)
            else:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue
            if len(buffer) >= BATCH_SIZE:
                try:
                    translated_abstracts = translate_abstracts(buffer)
                    for e, translated_abs in zip(entries_buffer, translated_abstracts):
                        e["abstract"] = translated_abs
                        fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                except Exception as ex:
                    print(f"Batch failed: {str(ex)[:200]}...")
                    for e in entries_buffer:
                        fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                finally:
                    buffer.clear()
                    entries_buffer.clear()
            processed += 1
        # Final batch
        if buffer:
            try:
                translated_abstracts = translate_abstracts(buffer)
                for e, translated_abs in zip(entries_buffer, translated_abstracts):
                    e["abstract"] = translated_abs
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            except Exception as ex:
                print(f"Final batch failed: {str(ex)[:200]}...")
                for e in entries_buffer:
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
    print("\nTranslation complete. Output saved to:", output_file)

if __name__ == "__main__":
    main()
