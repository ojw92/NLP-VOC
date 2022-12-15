-- 1) Check for empty cells in S22Ultra.Flair & update with 'None'
-- 2) Update all Samsung Community & Android Central sites' flair value in galaxys10 table as 'None'


-- 1)
/*
SELECT Title, Content, created_time, Flair
FROM S22Ultra
WHERE Flair IS '' AND			-- also try for IS NULL, depending on how NULL value was saved as during webscraping process
		--post_detail_link like '%www.reddit.com/r/GalaxyS22%' AND
		
		(substr(created_time, 7,4) ||	-- 4 characters starting from str index 7 (index starts at 1)
		 substr(created_time, 1,2) ||
		 substr(created_time, 4,2))
BETWEEN '20220000' and '20221111';		

UPDATE S22Ultra
SET Flair = 'None'
WHERE Flair IS '' AND
		--post_detail_link like '%www.reddit.com/r/GalaxyS22%' AND
		
		(substr(created_time, 7,4) ||
		 substr(created_time, 1,2) ||
		 substr(created_time, 4,2))
BETWEEN '20220000' and '20221107';
*/

-- 2)
/*
-- check for empty('') flair cells in Samsung Community's and Android Central's S22 Ultra VOC, fill with 'None' since no flair exists/necessary for those sites
SELECT title, description, sync, post_detail_link, sync_time, flair
FROM galaxys10
WHERE flair IS NULL AND
		(post_detail_link like '%community.samsung.com/t5/Galaxy-S22%' OR
		post_detail_link like '%androidcentral.com/samsung-galaxy-s22-s22-%') AND
		
		(substr(sync_time, 7,4) ||	-- 4 characters starting from str index 7 (index starts at 1)
		 substr(sync_time, 1,2) ||
		 substr(sync_time, 4,2))
BETWEEN '20220000' and '20221114';


UPDATE galaxys10
SET flair = 'None'
WHERE flair IS NULL AND
		(post_detail_link like '%community.samsung.com/t5/Galaxy-S22%' OR
		post_detail_link like '%androidcentral.com/samsung-galaxy-s22-s22-%') AND
		
		(substr(sync_time, 7,4) ||
		 substr(sync_time, 1,2) ||
		 substr(sync_time, 4,2))
BETWEEN '20220000' and '20221114';


SELECT title, description, sync, post_detail_link, sync_time, flair
FROM galaxys10
WHERE 
		(post_detail_link like '%community.samsung.com/t5/Galaxy-S22%' OR
		post_detail_link like '%androidcentral.com/samsung-galaxy-s22-s22-%') AND
		
		(substr(sync_time, 7,4) ||	-- 4 characters starting from str index 7 (index starts at 1)
		 substr(sync_time, 1,2) ||
		 substr(sync_time, 4,2))
BETWEEN '20220000' and '20221114';
*/


