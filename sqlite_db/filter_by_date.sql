-- filter_by_date.sql

-- filter VOC by date range
-- modify lines 10-13 in WHERE and line 18 in BETWEEN

-- https://stackoverflow.com/questions/4428795/sqlite-convert-string-to-date
-- SQLite doesn't have a date type. Need to do string comparison after converting : dd/MM/yyyy --> yyyyMMdd

-- concatenate yyyy, mm, and dd
SELECT title, description, sync, post_detail_link, sync_time
FROM galaxys10
WHERE sync like '%n%' AND post_detail_link like '%ATT/comments%' AND                                        -- filter by sync label & url

            --(title like '%watch4%' OR title like '%watch 4%' OR title like '%gw4%' OR               -- filter by keywords in title & description
            --description like '%watch4%' OR description like '%watch 4%' OR description like '%gw4%') AND

            (substr(sync_time, 7,4) ||    -- 4 characters starting from str index 7 (index starts at 1)
             substr(sync_time, 1,2) ||
             substr(sync_time, 4,2))
BETWEEN '20221101' and '20221107';                                                                                                        -- filter by date range
-- ORDER BY galaxys10.sync_time DESC     -- DESC shows newest VOC first


-- Update values at select columns
--UPDATE galaxys10
--SET flair = 'None'
--WHERE post_detail_link like '%https://forum.xda-developers.com%'


-- Notes:
--SELECT datetime('now');   -- or GetDate() in SQL Server(2008), Azure SQL DB/Data Warehouse, Parallel Data Warehouse