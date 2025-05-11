select distinct patient_type from patient_info 
select * from patient_info where patient_type is null
 
select * from user_info

--final join
select p.patient_id, p.first_name, p.last_name, p.date_of_birth, postal_code, user_type, registered_at, country, referral_group, veteran, ethnicity, race, city, state, dark_mode, p.gender, patient_type, patient_sub_type, external_id 
--,isr.immediate_symptoms_resulting
,hht.head_hit_count
,ntbi.has_tbi_before
from user_info u 
inner join patient_info p on u.first_name=p.first_name and u.last_name = p.last_name
inner join goals g on p.patient_id = g.patient_id --no col added yet
--inner join immediate_symptoms_resulting isr on isr.patient_id = p.patient_id
left join (select  count(head_hit_location) head_hit_count, patient_id from incident_head_hit_location hht group by patient_id ) hht on p.patient_id = hht.patient_id
left join nontbi_condition ntbi on ntbi.patient_id = p.patient_id
left join (select  count(therapies) therapy_cnt, patient_id from patient_therapies  group by patient_id ) ppt on p.patient_id = ppt.patient_id
 
select * from user_info --1375
select * from patient_info --1240
 
select count(*) a from goals group by patient_id  having count(patient_id) >1
 
select * from immediate_symptoms_resulting --mul rows for 1 patient
 
select * from incident_head_hit_location order by patient_id --
select count(*) a from incident_head_hit_location group by patient_id  having count(patient_id) >1
 
 
select * from new_resulting_factors where patient_about_id='9bb9fc04-9cc6-415f-a381-ddca289367cf' --
select count(*) a, patient_about_id from new_resulting_factors group by patient_about_id  having count(patient_about_id) >1
 
 
select * from nontbi_condition
select count(*) a from nontbi_condition group by patient_id  having count(patient_id) >1
 
 
select * from patient_therapies where patient_id is not null
select count(*) a from patient_therapies group by patient_id  having count(patient_id) >1
 
select * from register_tracking_list
 
----------------------------------------------------------------
--------------------------------------------------------------------
select * from registered_sdoh where patient_about_id in ('71de7056-c555-480d-8faf-6bc353e67d71')
 
select * from symptom_details where patient_id='1b0755ec-e30c-4477-8e62-c1330829316a'
select count(*) a, patient_id from symptom_details group by patient_id , symptom_id having count(patient_id) >1
 
 
select distinct count(patient_id)  from tbi_incident
select * from tbi_incident
 
select * from tracking_list

select * from goals
 
-----------------------------------------------------------------------------------------------------------------------------------------
 
select distinct p.patient_id from immediate_symptoms_resulting i inner join patient_info p on p.patient_id = i.patient_id -----useless
 
 
select * from experienced_sign
 
select * from head_hit_location
 
select * from patient_types
 
select * from symptom_list

select * from symptom_details

select * from new_resulting_factors
 
select * from tbi_from
 
select * from therapies_list
 
select * from user_info where user_type='caregiver' ---148
select * from user_info where user_type='patient' --- 1003
select * from user_info where user_type='Therapist' ---171
select * from user_info where user_type='provider'---40

---------------------------------------------------------------------- VIEW -----------------------------------------------------------------------
 
SELECT * FROM vw_patient_demographics_summary

