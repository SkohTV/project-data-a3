from datetime import datetime

# Chaîne de caractères à convertir
date_str = "2025-06-11t19%3a41"

# Remplacer le "t" par "T" et le "%3a" par ":"
date_str = date_str.replace("t", "T").replace("%3a", ":")

# Parser la chaîne en un objet datetime
date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M")

# Afficher l'objet datetime
print(date_obj)