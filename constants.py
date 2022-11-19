import os


MODEL_CACHE = "diffusers-cache"
TRANSLATOR_MODEL_CACHE = "translator-model-cache"
TRANSLATOR_TOKENIZER_CACHE = "translator-tokenizer-cache"

LANG_TO_ID = {
  "AFRIKAANS": "afr_Latn",
  "ALBANIAN": "als_Latn",
  "ARABIC": "arb_Arab",
  "ARMENIAN": "hye_Armn",
  "AZERBAIJANI": "azj_Latn",
  "BASQUE": "eus_Latn",
  "BELARUSIAN": "bel_Cyrl",
  "BENGALI": "ben_Beng",
  "BOKMAL": "nob_Latn",
  "BOSNIAN": "bos_Latn",
  "CATALAN": "cat_Latn",
  "CHINESE": "zho_Hans",
  "CROATIAN": "hrv_Latn",
  "CZECH": "ces_Latn",
  "DANISH": "dan_Latn",
  "DUTCH": "nld_Latn",
  "ENGLISH": "eng_Latn",
  "ESPERANTO": "epo_Latn",
  "ESTONIAN": "est_Latn",
  "FINNISH": "fin_Latn",
  "FRENCH": "fra_Latn",
  "GANDA": "lug_Latn",
  "GEORGIAN": "kat_Geor",
  "GERMAN": "deu_Latn",
  "GREEK": "ell_Grek",
  "GUJARATI": "guj_Gujr",
  "HEBREW": "heb_Hebr",
  "HINDI": "hin_Deva",
  "HUNGARIAN": "hun_Latn",
  "ICELANDIC": "isl_Latn",
  "INDONESIAN": "ind_Latn",
  "IRISH": "gle_Latn",
  "ITALIAN": "ita_Latn",
  "JAPANESE": "jpn_Jpan",
  "KAZAKH": "kaz_Cyrl",
  "KOREAN": "kor_Hang",
  "LATVIAN": "lvs_Latn",
  "LITHUANIAN": "lit_Latn",
  "MACEDONIAN": "mkd_Cyrl",
  "MALAY": "zsm_Latn",
  "MAORI": "mri_Latn",
  "MARATHI": "mar_Deva",
  "MONGOLIAN": "khk_Cyrl",
  "NYNORSK": "nno_Latn",
  "PERSIAN": "pes_Arab",
  "POLISH": "pol_Latn",
  "PORTUGUESE": "por_Latn",
  "PUNJABI": "pan_Guru",
  "ROMANIAN": "ron_Latn",
  "RUSSIAN": "rus_Cyrl",
  "SERBIAN": "srp_Cyrl",
  "SHONA": "sna_Latn",
  "SLOVAK": "slk_Latn",
  "SLOVENE": "slv_Latn",
  "SOMALI": "som_Latn",
  "SOTHO": "nso_Latn",
  "SPANISH": "spa_Latn",
  "SWAHILI": "swh_Latn",
  "SWEDISH": "swe_Latn",
  "TAGALOG": "tgl_Latn",
  "TAMIL": "tam_Taml",
  "TELUGU": "tel_Telu",
  "THAI": "tha_Thai",
  "TSONGA": "tso_Latn",
  "TSWANA": "tsn_Latn",
  "TURKISH": "tur_Latn",
  "UKRAINIAN": "ukr_Cyrl",
  "URDU": "urd_Arab",
  "VIETNAMESE": "vie_Latn",
  "WELSH": "cym_Latn",
  "XHOSA": "xho_Latn",
  "YORUBA": "yor_Latn",
  "ZULU": "zul_Latn",
}

TASKS_SWINIR = {
  'Real-World Image Super-Resolution-Medium': 'real_sr',
  'Real-World Image Super-Resolution-Large': 'real_sr',
  'Grayscale Image Denoising': 'gray_dn',
  'Color Image Denoising': 'color_dn',
  'JPEG Compression Artifact Reduction': 'jpeg_car'
}

MODEL_DIR_SWINIR = 'experiments/pretrained_models'

MODELS_SWINIR = {
  'real_sr': {
    "medium": os.path.join(MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"),
    "large": os.path.join(MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"),
  },
  'gray_dn': {
      15: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
      25: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
      50: os.path.join(MODEL_DIR_SWINIR, '004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth')
  },
  'color_dn': {
      15: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth'),
      25: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth'),
      50: os.path.join(MODEL_DIR_SWINIR, '005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth')
  },
  'jpeg_car': {
      10: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth'),
      20: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth'),
      30: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth'),
      40: os.path.join(MODEL_DIR_SWINIR, '006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth')
  }
}