Handwritten Spelling Error Recognition and Correction in Tamil Language: A Context-Aware Approach
Authors
Hari Krishnan J – 21MID0090, VIT Vellore (hariskrishnan.j2021@vitstudent.ac.in)
Arjun S – 21MID0088, VIT Vellore (arjun.s2021c@vitstudent.ac.in)
Shiny S – 21MID0079, VIT Vellore (shiny.2021@vitstudent.ac.in)
Abstract
Correcting spelling errors in handwritten Tamil text presents considerable challenges due to its intricate character set. Handwriting styles vary significantly, and words with similar sounds but different spellings further complicate accurate recognition.

This project integrates neural networks and natural language processing (NLP) techniques to develop a context-aware spelling correction system for handwritten Tamil text. The solution aims to:

Detect and correct spelling errors.
Understand the contextual significance of words.
Address challenges like character similarity and ambiguous handwriting.
By leveraging deep learning and NLP, this model ensures improved accuracy in handwritten Tamil text recognition and correction.

Keywords: Handwritten text, Neural networks, Natural language processing, Tamil script, Spelling correction.

Introduction
Tamil, one of the oldest Dravidian languages, has a complex script with 247 characters comprising:

12 vowels (உயிர் எழுத்து)
18 consonants (மெய் எழுத்து)
1 special letter (ஆயுத எழுத்து)
Due to these complexities, even minor variations in letters can lead to incorrect words, making spelling errors common. This is further exacerbated by:

Variability in handwriting styles.
Decreasing proficiency in written Tamil, even among fluent speakers.
The lack of automated tools for effective Tamil spelling correction.
This project digitizes handwritten Tamil text using Unicode and applies machine learning models to detect and correct errors based on context. The goal is to provide a self-learning tool for students and language enthusiasts, enabling them to write Tamil accurately without external assistance.

Project Structure
📂 Dataset & Preprocessing

class_to_unicode.csv – Mapping of handwritten Tamil characters to Unicode.
class_unicode_mapper.py – Converts handwritten text to Unicode.
word_segmentation.py – Splits handwritten input into meaningful word segments.
letter_segmentation.py – Extracts individual Tamil letters from words.
📂 Model & Algorithms

channel_model.py – Implements the spelling correction model.
combined_model.py – Merges multiple correction approaches.
com_1_edit_distance.py – Computes edit distance to find the most probable corrections.
compute_edit_distance.ipynb – Jupyter notebook for analyzing edit distances.
compute_error.ipynb – Evaluates model accuracy.
📂 Handwriting Recognition & Correction

tamil_character_recognition.py – Identifies Tamil characters from handwritten input.
sentence_recognition.ipynb – Processes entire sentences for corrections.
main.py – Main execution script.
📂 Miscellaneous

text.txt – Sample text for testing.
Installation & Setup
Prerequisites
Ensure you have the following installed:

Python 3.x
TensorFlow / PyTorch
OpenCV
NumPy
Pandas
Installation
Clone this repository:
bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the model:
bash
Copy
Edit
python main.py
Usage
Provide handwritten Tamil text as input.
The system converts it to digital form using Unicode.
The model identifies and corrects spelling errors based on context.
Outputs the corrected Tamil text.
Future Enhancements
🔹 Improve handwriting recognition using larger datasets.
🔹 Enhance contextual accuracy for better spelling correction.
🔹 Develop a web or mobile interface for broader accessibility.

Contributing
Contributions are welcome! If you’d like to improve the project:

Fork the repository
Create a new branch
Submit a pull request
License
This project is licensed under the MIT License – feel free to modify and distribute.
