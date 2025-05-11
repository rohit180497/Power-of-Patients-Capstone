-----------####################################   PATIENT DEMOGRAPHICS ########################################-----------------
--user_info
--patient_info
--goals
--incident_head_hit_location
--nontbi_condition
--patient_therapies
 
select p.patient_id, p.first_name, p.last_name, p.date_of_birth, postal_code, user_type, registered_at, country, referral_group, veteran, ethnicity, race, city, state, dark_mode, p.gender, patient_type, patient_sub_type, external_id 
--,isr.immediate_symptoms_resulting
--,hht.head_hit_count
,ntbi.has_tbi_before
from user_info u 
inner join patient_info p on u.first_name=p.first_name and u.last_name = p.last_name
inner join goals g on p.patient_id = g.patient_id --no col added yet
--inner join immediate_symptoms_resulting isr on isr.patient_id = p.patient_id
--left join (select  count(head_hit_location) head_hit_count, patient_id from incident_head_hit_location hht group by patient_id ) hht on p.patient_id = hht.patient_id
left join nontbi_condition ntbi on ntbi.patient_id = p.patient_id
--left join (select  count(therapies) therapy_cnt, patient_id from patient_therapies  group by patient_id ) ppt on p.patient_id = ppt.patient_id
 
 
-----------####################################   TBI INCIDENT ON REGISTRATION ########################################-----------------
--tbi_incident
--immediate_symptoms_resulting
--register_tracking_list
--symptom_list
--registered_sdoh
--therapies_list
--patient_therapies
 
-- TBI DETAILS AT THE TIME OF INJURY
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
 
-- worst TOP 3 SYMPTOMS 
select distinct rtl.patient_id,rtl.symptom_id, sl.id, sl.category, sl.subcategory,sl.factor from register_tracking_list rtl inner join symptom_list sl on sl.id = rtl.symptom_id order by patient_id
--register_tracking_list HAS LIST OF ALL FACTORS FROM ALL CATEGORIES(OLD VER) NEW VERSION AHS TOP 3 SYMPTOMS ONLY . IT IS FILLED DURING REGISTRATION. HAS DUPLICATEs. handled using distinct
 
-- ALL RECORDED MEDICAL SYMPTOMS
select distinct tl.patient_id, tl.symptom_id, sl.id, sl.category, sl.subcategory, sl.factor, sl.prime from tracking_list tl inner join symptom_list sl on sl.id = tl.symptom_id order by patient_id
 
-- ALL SDOH RECORDS  AT THE TIME OF REGISTRATION
SELECT distinct patient_about_id AS patient_id, symptom_id, category, subcategory, factor FROM registered_sdoh order by patient_about_id
 
-- ALL THERAPIES AT THE TIME OF REGISTRATION
SELECT DISTINCT patient_id, therapies_id, therapies, category FROM patient_therapies ORDER BY patient_id
--THIS TABLE NEEDS CLEANING- CATEGORY COL- REPLACE NULL WITH 'OTHER'
 
-----------####################################   SYMPTOM LOGS OVER TIME ########################################-----------------
--registered_factor
--new_resulting_factors
 
SELECT  count(*) FROM new_resulting_factors WHERE logged_at IS NOT NULL AND logged_at !='NULL' --  ORDER BY patient_about_id DESC
 
select distinct patient_about_id,symptom_date, logged_at, severity, category, subcategory, had_symptom, factor--, convert logged_at 
from new_resulting_factors 
WHERE logged_at IS NOT NULL AND logged_at !='NULL'
 
select count( *) from new_resulting_factors WHERE logged_at IS NOT NULL and logged_at !='NULL' --and severity!='null' 
-- severity , logged_at, data type correction
select distinct logged_at from new_resulting_factors order by logged_at desc  -- logged at has text values. need to be removed
 
-- 