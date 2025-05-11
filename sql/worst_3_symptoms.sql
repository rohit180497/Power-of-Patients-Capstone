select
    distinct rtl.patient_id,
    rtl.symptom_id,
    sl.id,
    sl.category,
    sl.subcategory,
    sl.factor
from
    register_tracking_list rtl
    inner join symptom_list sl on sl.id = rtl.symptom_id
order by
    patient_id