SELECT 
    p.patient_id,
    p.first_name,
    p.last_name,
    p.date_of_birth,
    u.postal_code,
    u.user_type,
    u.registered_at,
    u.country,
    u.referral_group,
    u.veteran,
    u.ethnicity,
    u.race,
    u.city,
    u.state,
    u.dark_mode,
    p.gender,
    p.patient_type,
    p.patient_sub_type,
-----    p.external_id,
    hht.head_hit_count,
    ntbi.has_tbi_before
    -- isr.immediate_symptoms_resulting (uncomment if needed)
FROM user_info u
INNER JOIN patient_info p 
    ON u.first_name = p.first_name 
    AND u.last_name = p.last_name
INNER JOIN goals g 
    ON p.patient_id = g.patient_id
-- LEFT JOIN immediate_symptoms_resulting isr 
--     ON isr.patient_id = p.patient_id
LEFT JOIN (
    SELECT 
        COUNT(head_hit_location) AS head_hit_count, 
        patient_id
    FROM incident_head_hit_location
    GROUP BY patient_id
) hht ON p.patient_id = hht.patient_id
LEFT JOIN nontbi_condition ntbi 
    ON ntbi.patient_id = p.patient_id
LEFT JOIN (
    SELECT 
        COUNT(therapies) AS therapy_cnt, 
        patient_id
    FROM patient_therapies
    GROUP BY patient_id
) ppt ON p.patient_id = ppt.patient_id;