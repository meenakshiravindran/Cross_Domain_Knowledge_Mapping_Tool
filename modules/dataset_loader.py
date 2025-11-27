import pandas as pd
import io
from modules.cache import cache



def load_dataset(uploaded_file):
    """Load and preprocess dataset from uploaded file."""
    try:
        # ---- CSV ----
        if uploaded_file.name.lower().endswith('.csv'):
            data = pd.read_csv(uploaded_file)
            cache.add_log('INFO', f"Loaded CSV file '{uploaded_file.name}' with {len(data)} records")

        # ---- TXT ----
        elif uploaded_file.name.lower().endswith('.txt'):
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            data = pd.DataFrame({'body': lines})
            cache.add_log('INFO', f"Loaded TXT file '{uploaded_file.name}' with {len(data)} lines")

        else:
            raise ValueError("Unsupported file format. Please upload CSV or TXT.")

        # Preprocess and return
        data = preprocess_data(data)
        return data

    except Exception as e:
        cache.add_log('ERROR', f"Error loading dataset: {str(e)}")
        raise e



def preprocess_data(data):
    """Preprocess dataset by identifying a text column and cleaning."""
    try:
        text_column = None

        # Most common expected names
        for col in ['body', 'text', 'content', 'description']:
            if col in data.columns:
                text_column = col
                break

        # If still not found, pick first object-type column
        if text_column is None:
            for col in data.columns:
                if data[col].dtype == 'object':
                    text_column = col
                    break

        # If still not found → dataset invalid
        if text_column is None:
            raise ValueError("No valid text column found in the dataset.")

        # Ensure dataset has "body" column
        if text_column != 'body':
            data['body'] = data[text_column]

        # Clean text data
        data['body'] = data['body'].fillna('').astype(str)
        data = data[data['body'].str.strip() != '']

        cache.add_log('INFO', f"Preprocessed dataset: {len(data)} valid text records")
        return data

    except Exception as e:
        cache.add_log('ERROR', f"Preprocessing failed: {str(e)}")
        raise e
