from normalization import normalize_record


TEST_CASES = [
    {
        "name": "ABC Healthcare, Pvt. Ltd.",
        "address": "88 Olive Circle, Lebanon, TN",
        "country": "US",
    },
    {
        "name": "Pinnacle Health Union Corp.",
        "address": "123 Main Road, Mumbai, Maharashtra",
        "country": "India",
    },
    {
        "name": "महाकाल ट्रेडर्स Pvt. Ltd.",
        "address": "F/403, Sahaj Residency, Mumbai",
        "country": "India",
    },
    {
        "name": "Middletown Pédiatric Dentistry Center Ltd",
        "address": "123 Rue de Paris, Lyon",
        "country": "France",
    },
]


print("=" * 70)
print("NORMALIZATION TEST")
print("=" * 70)


for i, case in enumerate(TEST_CASES, start=1):

    result = normalize_record(
        case["name"],
        case["address"],
        case["country"]
    )

    print(f"\nCASE {i}")

    print(
        "Original name:",
        case["name"]
    )

    print(
        "Normalized name:",
        result["business_name_normalized"]
    )

    print(
        "Name without suffix:",
        result["business_name_without_suffix"]
    )

    print(
        "Legal suffix:",
        result["legal_suffix"]
    )

    print(
        "Original address:",
        case["address"]
    )

    print(
        "Normalized address:",
        result["business_address_normalized"]
    )

    print(
        "Name script:",
        result["name_script"]
    )

    print(
        "Address script:",
        result["address_script"]
    )

    print("-" * 70)