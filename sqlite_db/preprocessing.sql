-- make the changes
-- exynos, uk, canada(ian), german(y), australia(n)/austria(n), india(n)
UPDATE galaxys10 SET sync = 'N' WHERE sync_time like '%09/18/2023%' AND (sync like '%y%' OR sync like '%t%') AND
(title like '%exy%' OR description like '%exy%' OR flair like '%exy%'
OR title like '%uk%' OR description like '%uk%'
OR title like '%euro%' OR description like '%euro%'
OR title like '%vodafone%' OR description like '%vodafone%'  -- British carrier
OR title like '%korea%' OR description like '%korea%'
OR title like '%china%' OR description like '%china%' OR title like '%chinese%' OR description like '%chinese%'
OR title like '%hong kong%' OR description like '%hong kong%' OR title like '%hongkong%' OR description like '%hongkong%'
OR title like '%taiwan%' OR description like '%taiwan%'
OR title like '%canad%' OR description like '%canad%'
OR title like '%toronto%' OR description like '%toronto%'
OR title like '%telus%' OR description like '%telus%'  -- Canadian carrier
OR title like '%freedom mobile%' OR description like '%freedom mobile%'  -- Canadian carrier
OR title like '%jio%' OR description like '%jio%'  -- JIO is Indian carrier
OR title like '%airtel%' OR description like '%airtel%'  -- Airtel is Indian carrier
OR title like '%deutsche%' OR description like '%deutsche%'  -- Deutsche Telekom is German carrier
OR title like '%hungar%' OR description like '%hungar%'
OR title like '%romania%' OR description like '%romania%'
OR title like '%austr%' OR description like '%austr%'
OR title like '%optus%' OR description like '%optus%'  -- Australian carrier
OR title like '%czech%' OR description like '%czech%'
OR title like '%denmark%' OR description like '%denmark%'
OR title like '%finland%' OR description like '%finland%'
OR title like '%netherl%' OR description like '%netherl%'
OR title like '%german%' OR description like '%german%'
OR title like '%india%' OR description like '%india%'
OR title like '%italy%' OR description like '%italy%'
OR title like '%france%' OR description like '%france%'
OR title like '%poland%' OR description like '%poland%'
OR title like '%New Zealand%' OR description like '%New Zealand%'
OR title like '%qatar%' OR description like '%qatar%'
OR title like '%dubai%' OR description like '%dubai%'
OR title like '%UAE%' OR description like '%UAE%'
OR title like '%United Arab Emirates%' OR description like '%United Arab Emirates%'
OR title like '%indonesia%' OR description like '%indonesia%'
OR title like '%philippines%' OR description like '%philippines%'
OR title like '%malay%' OR description like '%malay%'
OR title like '%macedonia%' OR description like '%macedonia%'
OR title like '%kuwait%' OR description like '%kuwait%'
OR title like '%mexic%' OR description like '%mexic%'
OR title like '%argentin%' OR description like '%argentin%'
OR title like '%colombia%' OR description like '%colombia%'
OR title like '%hondura%' OR description like '%hondura%'
OR title like '%guatemala%' OR description like '%guatemala%'
OR title like '%brazil%' OR description like '%brazil%'
OR title like '%portug%' OR description like '%portug%'
OR title like '%Spain%' OR description like '%Spain%'
OR title like '%israel%' OR description like '%israel%'
OR title like '%luxembourg%' OR description like '%luxembourg%'
OR title like '%beirut%' OR description like '%beirut%'
OR title like '%bangladesh%' OR description like '%bangladesh%'
OR title like '%bangkok%' OR description like '%bangkok%'
OR title like '%singapor%' OR description like '%singapor%'
OR title like '%egypt%' OR description like '%egypt%'
OR title like '%armenia%' OR description like '%armenia%'
OR title like '%London%' OR description like '%London%'
OR title like '%£%' OR description like '%£%' OR title like '%€%' OR description like '%€%' OR title like '%₹%' OR description like '%₹%'

OR title like '%beta%' OR description like '%beta%'
OR title like '%ui6%' OR description like '%ui6%'
OR title like '%ui 6%' OR description like '%ui 6%'

OR title like '%tab6%' OR description like '%tab6%' OR title like '%tab 6%' OR description like '%tab 6%'
OR title like '%tabs6%' OR description like '%tabs6%' OR title like '%tab s6%' OR description like '%tab s6%'
OR title like '%tab7%' OR description like '%tab7%' OR title like '%tab 7%' OR description like '%tab 7%'
OR title like '%tabs7%' OR description like '%tabs7%' OR title like '%tab s7%' OR description like '%tab s7%');

-- show results
--SELECT * FROM galaxys10
--WHERE sync_time like '%8/24/2022%'
--AND sync like '%N%'
--AND title like '%beta%'


--SELECT *
--FROM galaxys10;

-- https://www.sqlite.org/lang_datefunc.html
--SELECT strftime('%m/%d/%Y %H:%M:%S', datetime('now'), '-4 hours'); --shows UK time for some reason, so -4 hr
--SELECT strftime('%m/%d/%Y', datetime('now'), '-1 day', '-4 hours');

--UPDATE galaxys10 SET sync = 'N'
--WHERE sync_time like strftime('%m/%d/%Y', datetime('now'), '-1 day', '-4 hours')
--AND sync like '%y%'
--AND title like '%beta%';


-- https://www.w3schools.com/sql/sql_like.asp      character matching for "LIKE" operator
--work in progress (doesn't seem to be a way, according to: https://stackoverflow.com/questions/6526949/like-this-or-that-and-something-else-not-working)
-- exynos, uk, canada(ian), german(y), australia(n)/austria(n), india(n)
--UPDATE galaxys10 SET sync = 'N' WHERE sync_time like '%8/26/2022%' 
--AND sync like ('%y%' OR '%t%') 
--AND (title OR description) like ('%exy%' OR '%uk%' OR '%canad%' OR '%austr%' OR '%german%' OR '%india%');

--OR (title OR description) like ('%beta%' OR '%ui5%' OR '%ui 5%');

