from lingua import Language
import time
from transformers import pipeline
from .constants import LANG_TO_FLORES

target_lang_score_max = 0.8
target_lang = Language.ENGLISH
target_lang_flores = LANG_TO_FLORES[target_lang.name]

def translate_text(text, flores_200_code, model, tokenizer, detector, label):
    if text == "":
        print(f"-- {label} - No text to translate, skipping --")
        return ""
    startTimeTranslation = time.time()
    translated_text = ""
    text_lang_flores = target_lang_flores
    
    if flores_200_code != None:
        text_lang_flores = flores_200_code
        print(f'-- {label} - FLORES-200 code is given, skipping language auto-detection: "{text_lang_flores}" --')
    else:
        confidence_values = detector.compute_language_confidence_values(text)
        target_lang_score = None
        detected_lang = None
        detected_lang_score = None
        
        for index in range(len(confidence_values)):
            curr = confidence_values[index]
            if index == 0:
                detected_lang = curr[0]
                detected_lang_score = curr[1]
            if curr[0] == Language.ENGLISH:
                target_lang_score = curr[1]
                
        if detected_lang is not None and detected_lang != target_lang and (target_lang_score is None or target_lang_score < target_lang_score_max) and LANG_TO_FLORES.get(detected_lang.name) is not None:
            text_lang_flores = LANG_TO_FLORES[detected_lang.name]
        
        if detected_lang is not None:
            print(f'-- {label} - Guessed text language: "{detected_lang.name}". Score: {detected_lang_score} --')
        
        print(f'-- {label} - Selected text language FLORES-200: "{text_lang_flores}" --')

    if text_lang_flores != target_lang_flores:
        translate = pipeline(
            'translation',
            model=model,
            tokenizer=tokenizer,
            src_lang=text_lang_flores,
            tgt_lang=target_lang_flores,
            device=0
        )
        translate_output = translate(text, max_length=500)
        translated_text = translate_output[0]['translation_text']
        print(f'-- {label} - Translated text is: "{translated_text}" --')
    else:
        translated_text = text
        print(f"-- {label} - Text is already in the correct language, no translation needed --")
    
    endTimeTranslation = time.time()
    print(f"-- {label} - Completed in: {(endTimeTranslation - startTimeTranslation) * 10} sec. --")
    
    return translated_text