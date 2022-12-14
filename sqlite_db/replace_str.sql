-- replace_str.sql

-- replace substrings in VOC description (delete redundant text that muddies data cleanliness)
      -- us.community.samsung.com's "Opens in new windowPDF Download..." text that gets scraped along with VOC
      -- &amp;#x200B; on Reddit posts should just be whitespace
      -- &amp;#37; on Reddit is supposed to be %
      -- AT&amp;T should just be AT&T
      -- &amp; on Reddit posts be just &

-- Use the lines below to display the target rows first
SELECT title, description, sync, post_detail_link, sync_time
FROM galaxys10
WHERE (sync like '%n%' OR sync like '%y%' OR sync like '%r%') AND
            description like '%at&amp;t%' AND
            --description like '%Opens in new windowPDF DownloadWord DownloadExcel DownloadPowerPoint DownloadDocument Download%' AND
            --post_detail_link like '%https://us.community.samsung.com%' AND
            
            (substr(sync_time, 7,4) ||    -- 4 characters starting from str index 7 (index starts at 1)
             substr(sync_time, 1,2) ||
             substr(sync_time, 4,2))
BETWEEN '20221107' and '20221109';


-- Then use these lines to replace/remove substring from the target rows
UPDATE galaxys10
SET description = REPLACE(description,'Opens in new windowPDF DownloadWord DownloadExcel DownloadPowerPoint DownloadDocument Download', '');
UPDATE galaxys10
SET description = REPLACE(description, '&amp;#x200B;', '');
UPDATE galaxys10
SET description = REPLACE(description, '&amp;#37;', '%');
UPDATE galaxys10
SET title = REPLACE(title, '&amp;#37;', '%');
UPDATE galaxys10
SET description = REPLACE(description, 'at&amp;t', 'AT&T');
UPDATE galaxys10
SET title = REPLACE(title, 'at&amp;t', 'AT&T');