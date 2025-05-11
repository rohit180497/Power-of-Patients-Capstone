select distinct patient_about_id,symptom_date, logged_at, severity, category, subcategory, had_symptom, factor--, convert logged_at 
from new_resulting_factors 
WHERE logged_at IS NOT NULL AND logged_at !='NULL'