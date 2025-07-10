from datasets import load_dataset
import pandas as pd

# Download and load train split (approx. 270k comments)
ds = load_dataset("breadlicker45/youtube-comments", split="train")

print(ds.column_names)
# Convert to DataFrame and rename
df = pd.DataFrame(ds)[["comment_text", "label"]]
df.columns = ["comment", "sentiment"]

# Convert numeric labels (0=negative,1=neutral,2=positive) to text
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment"] = df["sentiment"].map(label_map)
