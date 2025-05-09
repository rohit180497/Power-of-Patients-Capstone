select top 10 * from symptom_list

select top 10 * from symptom_details sd
inner join symptom_list sl on sl.id = sd.symptom_id

select top 10 * from head_hit_location

select top 10 * from new_resulting_factors

select top 10 * from user_info where user_id in ('6d3a2d37-6d0b-465e-aef6-a9d2ac6e2e23')

select * from new_resulting_factors as t1 ---where id in (29440)
where user_input_id in ('6d3a2d37-6d0b-465e-aef6-a9d2ac6e2e23')


select * from patient_therapies pt
inner join therapies_list tl on tl.id = pt.therapies_id

select top 10 * from patient_info where patient_id in ('a19d709f-1440-4cd0-9175-7b7368ecfdc3')

select top 10 * from registered_sdoh

select top 10 * from therapies_list

select  id, first_goal from goals where first_goal is not null

user_info