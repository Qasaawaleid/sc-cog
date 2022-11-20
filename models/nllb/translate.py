from lingua import Language
import time
from transformers import pipeline
from constants import LANG_TO_ID

eng_score_max = 0.9
target_lang = Language.ENGLISH
target_lang_id = LANG_TO_ID[target_lang.name]

def translate_text(text, model, tokenizer, detector, label):
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""
    startTimeTranslation = time.time()
    translated_text = ""
    text_lang_id = target_lang_id
    
    confidence_values = detector.compute_language_confidence_values(text)
    eng_value = None
    detected_lang = None
    detected_lang_score = None
    
    for index in range(len(confidence_values)):
        curr = confidence_values[index]
        if index == 0:
            detected_lang = curr[0]
            detected_lang_score = curr[1]
        if curr[0] == Language.ENGLISH:
            eng_value = curr[1]
            
    if detected_lang is not None and detected_lang != target_lang and (eng_value is None or eng_value < eng_score_max) and LANG_TO_ID.get(detected_lang.name) is not None:
        text_lang_id = LANG_TO_ID[detected_lang.name]
    
    if detected_lang is not None:
        print(f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --')
    print(f'-- {label} - Selected text language id: "{text_lang_id}" --')
    
    if text_lang_id != target_lang_id:
        translate = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang=text_lang_id,
            tgt_lang=target_lang_id,
            device=0
        )
        translate_output = translate(text, max_length=500)
        translated_text = translate_output[0]['translation_text']
        print(f'-- {label} - Translated text is: "{translated_text}" --')
    else:
        translated_text = text
        print(f"-- {label} - Text is already in the correct language, no translation needed --")
    
    endTimeTranslation = time.time()
    print(f"-- {label} - Completed in: {endTimeTranslation - startTimeTranslation} sec. --")
    
    return translated_text