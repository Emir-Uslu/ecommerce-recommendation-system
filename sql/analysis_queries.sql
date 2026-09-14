SELECT
    event,
    COUNT(*) AS event_count
FROM events
GROUP BY event
ORDER BY event_count DESC;


SELECT
    COUNT(DISTINCT visitorid) AS unique_users,
    COUNT(DISTINCT itemid) AS unique_items,
    COUNT(*) AS total_events
FROM events;


SELECT
    COUNT(*) FILTER (WHERE event = 'view') AS views,
    COUNT(*) FILTER (WHERE event = 'addtocart') AS add_to_carts,
    COUNT(*) FILTER (WHERE event = 'transaction') AS transactions,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE event = 'addtocart')
        / NULLIF(COUNT(*) FILTER (WHERE event = 'view'), 0),
        2
    ) AS view_to_cart_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE event = 'transaction')
        / NULLIF(COUNT(*) FILTER (WHERE event = 'addtocart'), 0),
        2
    ) AS cart_to_purchase_pct,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE event = 'transaction')
        / NULLIF(COUNT(*) FILTER (WHERE event = 'view'), 0),
        2
    ) AS view_to_purchase_pct
FROM events;


SELECT
    event,
    COUNT(DISTINCT visitorid) AS users
FROM events
GROUP BY event
ORDER BY users DESC;


SELECT
    itemid,
    COUNT(*) AS total_interactions
FROM events
GROUP BY itemid
ORDER BY total_interactions DESC
LIMIT 20;


SELECT
    itemid,
    COUNT(*) FILTER (WHERE event = 'view') AS views,
    COUNT(*) FILTER (WHERE event = 'addtocart') AS carts,
    COUNT(*) FILTER (WHERE event = 'transaction') AS purchases,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE event = 'transaction')
        / NULLIF(COUNT(*) FILTER (WHERE event = 'view'), 0),
        2
    ) AS purchase_conversion_pct
FROM events
GROUP BY itemid
HAVING COUNT(*) FILTER (WHERE event = 'view') >= 100
ORDER BY purchases DESC
LIMIT 20;


SELECT
    visitorid,
    COUNT(*) AS total_events,
    COUNT(DISTINCT itemid) AS unique_items,
    COUNT(*) FILTER (WHERE event = 'transaction') AS purchases
FROM events
GROUP BY visitorid
ORDER BY total_events DESC
LIMIT 20;


SELECT
    EXTRACT(
        HOUR FROM to_timestamp(timestamp / 1000.0)
    ) AS hour,
    COUNT(*) AS events
FROM events
GROUP BY hour
ORDER BY hour;


SELECT
    TO_CHAR(
        to_timestamp(timestamp / 1000.0),
        'Day'
    ) AS day_of_week,
    COUNT(*) AS events
FROM events
GROUP BY
    EXTRACT(
        DOW FROM to_timestamp(timestamp / 1000.0)
    ),
    day_of_week
ORDER BY
    EXTRACT(
        DOW FROM to_timestamp(timestamp / 1000.0)
    );


SELECT
    itemid,
    COUNT(*) FILTER (WHERE event = 'view') AS views,
    COUNT(*) FILTER (WHERE event = 'transaction') AS purchases,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE event = 'transaction')
        / NULLIF(COUNT(*) FILTER (WHERE event = 'view'), 0),
        2
    ) AS conversion_pct
FROM events
GROUP BY itemid
HAVING
    COUNT(*) FILTER (WHERE event = 'view') >= 500
ORDER BY conversion_pct ASC, views DESC
LIMIT 20;