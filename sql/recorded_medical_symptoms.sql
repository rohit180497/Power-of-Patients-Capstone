select
    distinct tl.patient_id,
    tl.symptom_id,
    sl.id,
    sl.category,
    sl.subcategory,
    sl.factor,
    sl.prime
from
    tracking_list tl
    inner join symptom_list sl on sl.id = tl.symptom_id
order by
    patient_id