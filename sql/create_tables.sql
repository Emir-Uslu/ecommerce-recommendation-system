CREATE INDEX IF NOT EXISTS idx_events_visitor
ON events(visitorid);

CREATE INDEX IF NOT EXISTS idx_events_item
ON events(itemid);

CREATE INDEX IF NOT EXISTS idx_events_event
ON events(event);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
ON events(timestamp);