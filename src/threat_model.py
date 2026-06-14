"""
threat_model.py
Операціоналізована модель загроз: класифікація 111 ознак датасету Vrbancic
за рівнем контролю атакувальника при крафтингу фішингового URL.

Логіка поділу:
  CONTROLLABLE      — лексичні/структурні ознаки самого рядка URL. Атакувальник
                      пише URL власноруч, тож контролює їх повністю й безкоштовно.
  SEMI_CONTROLLABLE — ознаки інфраструктури, які атакувальник може змінити, лише
                      якщо керує власним хостингом/DNS/сертифікатами (з певними
                      зусиллями та вартістю).
  UNCONTROLLABLE    — ознаки, що визначаються третіми сторонами або часом і не
                      піддаються довільному збуренню (вік домену, індексація
                      Google, час відповіді сервера, ASN тощо).

Саме UNCONTROLLABLE робить «наївну» необмежену атаку FGSM/PGD нереалістичною:
вона вільно змінює, напр., вік домену чи факт індексації в Google — чого реальний
атакувальник зробити не може.
"""

# Індекси відповідають порядку колонок у dataset_small.csv (0..110)

CONTROLLABLE = [
    # Лексика всього URL
    "qty_dot_url", "qty_hyphen_url", "qty_underline_url", "qty_slash_url",
    "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
    "qty_exclamation_url", "qty_space_url", "qty_tilde_url", "qty_comma_url",
    "qty_plus_url", "qty_asterisk_url", "qty_hashtag_url", "qty_dollar_url",
    "qty_percent_url", "qty_tld_url", "length_url",
    # Лексика домену
    "qty_dot_domain", "qty_hyphen_domain", "qty_underline_domain", "qty_slash_domain",
    "qty_questionmark_domain", "qty_equal_domain", "qty_at_domain", "qty_and_domain",
    "qty_exclamation_domain", "qty_space_domain", "qty_tilde_domain", "qty_comma_domain",
    "qty_plus_domain", "qty_asterisk_domain", "qty_hashtag_domain", "qty_dollar_domain",
    "qty_percent_domain", "qty_vowels_domain", "domain_length",
    "domain_in_ip", "server_client_domain",
    # Лексика каталогу
    "qty_dot_directory", "qty_hyphen_directory", "qty_underline_directory",
    "qty_slash_directory", "qty_questionmark_directory", "qty_equal_directory",
    "qty_at_directory", "qty_and_directory", "qty_exclamation_directory",
    "qty_space_directory", "qty_tilde_directory", "qty_comma_directory",
    "qty_plus_directory", "qty_asterisk_directory", "qty_hashtag_directory",
    "qty_dollar_directory", "qty_percent_directory", "directory_length",
    # Лексика імені файлу
    "qty_dot_file", "qty_hyphen_file", "qty_underline_file", "qty_slash_file",
    "qty_questionmark_file", "qty_equal_file", "qty_at_file", "qty_and_file",
    "qty_exclamation_file", "qty_space_file", "qty_tilde_file", "qty_comma_file",
    "qty_plus_file", "qty_asterisk_file", "qty_hashtag_file", "qty_dollar_file",
    "qty_percent_file", "file_length",
    # Лексика параметрів
    "qty_dot_params", "qty_hyphen_params", "qty_underline_params", "qty_slash_params",
    "qty_questionmark_params", "qty_equal_params", "qty_at_params", "qty_and_params",
    "qty_exclamation_params", "qty_space_params", "qty_tilde_params", "qty_comma_params",
    "qty_plus_params", "qty_asterisk_params", "qty_hashtag_params", "qty_dollar_params",
    "qty_percent_params", "params_length", "tld_present_params", "qty_params",
    "email_in_url",
    # Вибір сервісу скорочення
    "url_shortened",
]

SEMI_CONTROLLABLE = [
    "domain_spf",            # SPF-запис — за умови контролю над DNS
    "qty_nameservers",       # кількість NS — за умови self-hosted DNS
    "qty_mx_servers",        # кількість MX — за умови контролю над поштою
    "ttl_hostname",          # TTL — за умови контролю над DNS
    "tls_ssl_certificate",   # валідний TLS — нині доступні безкоштовні сертифікати
    "qty_redirects",         # кількість редиректів — конфігурується атакувальником
    "time_domain_expiration",# термін реєстрації — обирається при купівлі домену
]

UNCONTROLLABLE = [
    "time_response",          # час відповіді сервера — мережа/інфраструктура
    "asn_ip",                 # ASN — визначається хостинг-провайдером
    "time_domain_activation", # вік домену — час; не підробити без реальних витрат
    "qty_ip_resolved",        # к-сть IP — інфраструктура DNS
    "url_google_index",       # індексація URL у Google — рішення третьої сторони
    "domain_google_index",    # індексація домену в Google — рішення третьої сторони
]


def build_masks(feature_names):
    """
    Повертає три булеві маски (numpy) довжиною len(feature_names):
      mask_lexical  — лише CONTROLLABLE (найконсервативніша реалістична атака)
      mask_realistic— CONTROLLABLE + SEMI_CONTROLLABLE (атакувальник керує хостингом)
      mask_full     — усі ознаки (необмежена «наївна» атака)
    """
    import numpy as np
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    # Перевірка повноти класифікації
    classified = set(CONTROLLABLE) | set(SEMI_CONTROLLABLE) | set(UNCONTROLLABLE)
    missing = [n for n in feature_names if n not in classified]
    extra = [n for n in classified if n not in name_to_idx]
    if missing:
        raise ValueError(f"Некласифіковані ознаки: {missing}")
    if extra:
        raise ValueError(f"Класифіковано неіснуючі ознаки: {extra}")

    d = len(feature_names)
    mask_lexical = np.zeros(d, dtype=bool)
    mask_realistic = np.zeros(d, dtype=bool)
    mask_full = np.ones(d, dtype=bool)

    for n in CONTROLLABLE:
        mask_lexical[name_to_idx[n]] = True
        mask_realistic[name_to_idx[n]] = True
    for n in SEMI_CONTROLLABLE:
        mask_realistic[name_to_idx[n]] = True

    return {
        "lexical": mask_lexical,
        "realistic": mask_realistic,
        "full": mask_full,
    }


SUMMARY = {
    "controllable": len(CONTROLLABLE),
    "semi_controllable": len(SEMI_CONTROLLABLE),
    "uncontrollable": len(UNCONTROLLABLE),
    "total": len(CONTROLLABLE) + len(SEMI_CONTROLLABLE) + len(UNCONTROLLABLE),
}

if __name__ == "__main__":
    print("Феатур-таксономія (модель загроз):")
    for k, v in SUMMARY.items():
        print(f"  {k:18s}: {v}")
