# Query Analysis Service - Grafana Dashboard

Comprehensive monitoring dashboard for the Query Analysis microservice with 18 visualization panels.

## Dashboard Overview

The dashboard provides real-time monitoring and analytics for:
- Query volume and processing performance
- Intent classification distribution
- Cache performance metrics
- Entity extraction analytics
- Query expansion statistics
- Error rates and service health
- User activity tracking

## Dashboard Panels

### Performance Metrics (Row 1)
1. **Total Query Volume** - Real-time query count with thresholds
2. **Average Processing Time** - Mean latency across all queries
3. **Cache Hit Rate** - Percentage of cache hits vs misses
4. **Active Users** - Unique users in the last hour

### Trends & Patterns (Row 2)
5. **Query Volume Over Time** - Time series of queries per minute
6. **Processing Time Percentiles** - p50, p90, p95, p99 latencies

### Analytics (Row 3)
7. **Intent Distribution** - Pie chart of 8 intent categories
8. **Cache Performance** - Cache hits vs misses over time
9. **Top 10 Popular Queries** - Most frequent queries table

### Advanced Analytics (Rows 4-6)
10. **Entity Types Distribution** - Bar chart of extracted entities
11. **Query Expansion Performance** - Expansion requests and avg generated
12. **Error Rate** - Errors per second with alerting
13. **Service Health Status** - UP/DOWN indicator
14. **Language Distribution** - Detected languages pie chart
15. **Top Keywords (24h)** - Most frequent keywords table
16. **RabbitMQ Consumer Lag** - Queue processing backlog
17. **Cache Latency Comparison** - Performance with/without cache
18. **Query Complexity Score** - Distribution of query complexity

## Quick Start

### Access Dashboard

1. Start services: `docker-compose up -d`
2. Access Grafana: http://localhost:3000 (admin/admin)
3. Navigate to Dashboards → RAG System → Query Analysis

### Import Dashboard Manually

```bash
curl -X POST http://localhost:3000/api/dashboards/import \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d @query-analysis-dashboard.json
```

## Key Metrics to Monitor

- **Processing Time p95**: Should be < 500ms
- **Cache Hit Rate**: Target > 70%
- **Error Rate**: Should be < 1%
- **Query Volume**: Monitor for traffic spikes
- **Consumer Lag**: Should stay near 0

For full documentation, see the dashboard README in this directory.
