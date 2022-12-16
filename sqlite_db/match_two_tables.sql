-- 0) Replace all '' in S22Ultra.Flair with 'None'
-- 1) Match id from galaxys10 & S22Ultra and update flair(galaxys10) with Flair(S22Ultra)
-- 2) Match id from r/S22Ultra, r/GalaxyS22, and other subreddits whose posts mention 's22' in hyperlink (title)
-- 3) Match titles & started_by_time/created_time from 2 tables whose ids do not match


-- 0)
/*
-- Check which rows of S22Ultra have '' in Flair column
SELECT id, S22Ultra.title, Content, created_time, S22Ultra.Flair
FROM S22Ultra
WHERE S22Ultra.Flair IS '';

-- Update those rows to have 'None' instead of ''
UPDATE S22Ultra
SET Flair = 'None'
WHERE S22Ultra.Flair IS '';

-- Check to see if the rows were updated successfully
SELECT id, S22Ultra.title, Content, created_time, S22Ultra.Flair
FROM S22Ultra
WHERE S22Ultra.Flair IS '';
*/

-- 1) & 2) 
/*
SELECT galaxys10.title, started_by_time, post_id, galaxys10.flair,	 -- include table name, if ambiguous
		S22Ultra.Flair, id, created_time, S22Ultra.title
FROM galaxys10
INNER JOIN S22Ultra ON S22Ultra.id = galaxys10.post_id -- AND  S22Ultra.title = galaxys10.title
		-- 1) reddit.com/r/S22Ultra (8260), reddit.com/r/GalaxyS22 (5038), total (13298)
		--WHERE (post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%');

		-- 2) other reddit.com subreddits (3222), but only (2706 match ids from both tables)
		WHERE (post_detail_link like '%www.reddit.com%' AND post_detail_link like '%s22%'); --AND
		--	NOT (post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%');
		-- 3) posts with mismatched ids should be 516
		

-- add lines to update flair with Flair
	-- https://stackoverflow.com/questions/3845718/update-table-values-from-another-table-with-the-same-user-name
UPDATE galaxys10
SET
	flair = (SELECT S22Ultra.Flair
					FROM S22Ultra
					WHERE S22Ultra.id = galaxys10.post_id AND
						-- 1) reddit.com/r/S22Ultra (8260), reddit.com/r/GalaxyS22 (5038), total (13298)
						--	S22Ultra.title = galaxys10.title AND
						--	(post_detail_link like '%reddit.com/r/S22Ultra%' OR post_detail_link like '%reddit.com/r/GalaxyS22%')
						-- 2) other reddit.com subreddits (3222), but only (2706 match ids from both tables)
							(post_detail_link like '%www.reddit.com%' AND post_detail_link like '%s22%') --AND
						--	NOT (post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%')
							);

							
-- check the updated column
SELECT galaxys10.title, started_by_time, galaxys10.flair,
		S22Ultra.Flair, created_time, S22Ultra.title
FROM galaxys10
INNER JOIN S22Ultra ON S22Ultra.id = galaxys10.post_id -- AND  S22Ultra.title = galaxys10.title
WHERE
		-- 1) reddit.com/r/S22Ultra (8260), reddit.com/r/GalaxyS22 (5038), total (13298)
		--(post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%');

		-- 2) other reddit.com subreddits (3222), but only (2706 match ids from both tables)
			(post_detail_link like '%www.reddit.com%' AND post_detail_link like '%s22%'); --AND
			--	NOT (post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%');
*/

-- 3)
/*
-- Check which rows of reddit.com in galaxys10 have NULL in flair column after updating matching ids from S22Ultra
SELECT galaxys10.title, started_by_time, post_id, galaxys10.flair, post_detail_link
FROM galaxys10
WHERE galaxys10.flair IS NULL AND
		(post_detail_link like '%www.reddit.com%' AND post_detail_link like '%s22%') AND 
		(substr(sync_time, 7,4) ||
		 substr(sync_time, 1,2) ||
		 substr(sync_time, 4,2))
BETWEEN '20220000' and '20221114';
-- 3864 results
*/

/*
-- Now check for rows that have same titles. Check for similar started_by_time/created_time
SELECT galaxys10.title, started_by_time, post_id, galaxys10.flair,	 -- include table name, if ambiguous
		S22Ultra.Flair, id, created_time, S22Ultra.title
FROM galaxys10
INNER JOIN S22Ultra ON S22Ultra.id = galaxys10.post_id AND  S22Ultra.title = galaxys10.title
		-- 1) reddit.com/r/S22Ultra (8260), reddit.com/r/GalaxyS22 (5038), total (13298)
		WHERE (post_detail_link like '%www.reddit.com/r/GalaxyS22%' OR post_detail_link like '%www.reddit.com/r/S22Ultra%');
*/


-- ****** Technical problem with SQLite? ******
-- Why does updating all of galaxys10's Samsung Community & Android Central flairs to 'None', saving (& tried exiting as well) and then
-- updating galaxys10's Reddit flairs with S22Ultra's Reddit Flairs cause Samsung & Android Central cells to be undone? (go back to NULL)

