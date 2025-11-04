# Transfer Learning for Animal Sounds with YAMNet

Idenitfying how well Google's YAMNet transfers to classifying different types of sounds - in this case, environment recordings with animal sounds. Specifically, we test whether embeddings learned from **music** and **environmental sounds** can help recognize **animal sounds**.

**Team:**  
Oyku Cildir · Claudia Martinez  
MATH 80648A – Deep Learning Project (HEC Montréal, Fall 2025)

## Experiments

| Experiment | Training Data | Test Data | Purpose |
|-------------|----------------|------------|----------|
| Zero-shot | GTZAN | ESC-50 (animals) | Measure generalization without adaptation |
| Few-shot | GTZAN + few labeled ESC-50 samples | ESC-50 | Measure adaptability with limited labels |
| Full fine-tune | ESC-50 (full) | ESC-50 | Measure upper-bound performance |

## Datasets

- **GTZAN Music Genre Dataset** – [Kaggle link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
  → 1000 clips (30 s each, 10 genres)

- **ESC-50 (Animal subset)** – [GitHub link](https://github.com/karolpiczak/ESC-50)  
  → 2000 clips (5 s each, 10 animal categories)

All audio need to be resampled to **16 kHz mono**, chunked into ~1 s windows for YAMNet processing.

## Note on YAMNet Embeddings
Example: 
We have a clip of a 5 sec dog barking, we first would need to split this into 1s segments.
We would then pass each segment through YAMNet that would produce an output of 1024 numbers that describe different "micro-features" of the sound segment.
For a 5 second clip, the overall output will be a 5x1024 matrix.
When we aggregate those outputs, we will need to summarize them into **one single vector** --> we will do this by taking an avereage of the 5 vectors and creating one final 1024 dimension vector (aka. Mean Pooling).

## Note on Metrics to Calculate
Accuracy = Number of correct predictions / total # of samples
- It is useful as a first step to see if the model is working well.
- Not sufficient for transfer learning, since accuracy alone can hide poor performance on small or rare classes.

Macro F-1 Score is the macro average of --> 2x (precision x recall) / (precision + recall) (over all classes, C)
- This metric is calculated per class.
- Precision is the number of samples that were classified as X that indeed are X (of all clips the model predicted as “dog,” how many were actually dog?)
- Recall is the number of samples that belong to a class that were correctly classified (of all real dog clips, how many did the model correctly find?)
- Treats every class equally, no matter how many samples it has.
- Highlights whether the model is biased (e.g., good on “bird” but bad on “frog”).

Confusion Matrix: A full breakdown of which classes the model confuses with each other (rows = true classes, columns = predicted classes)
- Can be used to diagnose what kind of mistakes the model does
- 

## Workflow Summary
1. Preprocessing the input audio files
   - Resample the audio files to 16 kHz mono
   - Segment them into 1s windows so that YAMNet can process them
2. Extracting embeddings (see note on embeddings above)
   - 1024-D vector per 1s segment
   - Mean pooling to aggregate the segment level vectors for the whole clip
3. Train classifiers (see experiments section above)
   - Zero shot
   - Few shot
   - Full finetune
4. Evaluate Metrics
   - Accuracy
   - Macro-F1
   - Confusion matrix
   - Performance of F1 vs k for few-shot

## How to Run the Code
1. Create environment:
pip install -r requirements.txt

2. Preprocess and extract embeddings:
python src/data/preprocess_audio.py --config configs/base.yaml
python src/models/extract_embeddings.py --config configs/base.yaml

3. Run experiments
python src/train/train_classifier.py --config configs/base.yaml --mode zero_shot
python src/train/train_classifier.py --config configs/base.yaml --mode few_shot
python src/train/train_classifier.py --config configs/base.yaml --mode fine_tune
