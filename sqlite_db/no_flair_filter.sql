-- no_flair_filter

SELECT title, description, sync, post_detail_link, sync_time, flair
FROM galaxys10
WHERE flair IS NULL AND
            post_detail_link like '%www.reddit.com/r/GalaxyS22%' AND
            (substr(sync_time, 7,4) ||    -- 4 characters starting from str index 7 (index starts at 1)
             substr(sync_time, 1,2) ||
             substr(sync_time, 4,2))
BETWEEN '20220000' and '20221109';