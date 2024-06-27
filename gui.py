import tkinter as tk
from tkinter import messagebox
import checker_code


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function to check plagiarism
def check_plagiarism():
    text1 = text1_entry.get("1.0", tk.END).strip()
    text2 = text2_entry.get("1.0", tk.END).strip()

    if not text1 or not text2:
        messagebox.showwarning("Input Error", "Both text fields must be filled.")
        return

    text1_processed = checker_code.preprocess(text1)
    text2_processed = checker_code.preprocess(text2)

    similarity = checker_code.calculate_similarity(text1_processed, text2_processed)
    result_text = f"Cosine Similarity: {similarity:.2f}\n"

    threshold = 0.8
    if similarity > threshold:
        result_text += "Potential plagiarism detected."
    else:
        result_text += "No plagiarism detected."

    result_label.config(text=result_text)

# Create the main window
root = tk.Tk()
root.title("Plagiarism Checker")

# Create and place text fields
tk.Label(root, text="Text 1:").grid(row=0, column=0, padx=10, pady=10)
text1_entry = tk.Text(root, height=10, width=50)
text1_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Text 2:").grid(row=1, column=0, padx=10, pady=10)
text2_entry = tk.Text(root, height=10, width=50)
text2_entry.grid(row=1, column=1, padx=10, pady=10)

# Create and place the Check button
check_button = tk.Button(root, text="Check for Plagiarism", command=check_plagiarism)
check_button.grid(row=2, column=0, columnspan=2, pady=10)

# Create and place the result label
result_label = tk.Label(root, text="", fg="blue")
result_label.grid(row=3, column=0, columnspan=2, pady=10)

# Run the main loop
root.mainloop()