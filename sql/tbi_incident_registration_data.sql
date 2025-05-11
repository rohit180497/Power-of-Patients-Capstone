WITH ranked_incidents AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY patient_id 
               ORDER BY tbi_incident_date DESC
           ) AS rn
    FROM tbi_incident
)
SELECT 
    patient_id,
    tbi_incident_date,
    injury_from,
    head_hit_location,
    num_head_hit_location,
    total_tbi,
    immediate_symptoms_resulting,
    describe_event
FROM ranked_incidents
WHERE rn = 1;