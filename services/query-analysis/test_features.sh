#!/bin/bash

echo "========================================="
echo "Query Analysis Service - Feature Tests"
echo "========================================="
echo ""

echo "Test 1: Health Check"
echo "-------------------"
curl -s http://localhost:8101/health
echo ""
echo ""

echo "Test 2: ML-Based Intent Classification (Procedural Query)"
echo "----------------------------------------------------------"
curl -s -X POST http://localhost:8101/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I implement OAuth2 authentication in Python?", "user_id": "testuser1"}' \
  | jq '.analysis.intent'
echo ""

echo "Test 3: Query Expansion"
echo "----------------------"
curl -s -X POST http://localhost:8101/api/v1/expand \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}' \
  | jq '.expansion | {original_query, total_expansions, expansion_methods, sample_expansions: .all_expansions[:3]}'
echo ""

echo "Test 4: Cache Statistics"
echo "-----------------------"
curl -s http://localhost:8101/api/v1/cache/stats | jq '.stats'
echo ""

echo "Test 5: Comparative Intent (Testing Different Intent)"
echo "-----------------------------------------------------"
curl -s -X POST http://localhost:8101/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the difference between REST and GraphQL?", "user_id": "testuser1"}' \
  | jq '.analysis.intent'
echo ""

echo "Test 6: Wait 2 seconds for consumer processing..."
sleep 2
echo ""

echo "Test 7: User Query History"
echo "-------------------------"
curl -s "http://localhost:8101/api/v1/history/testuser1?limit=5" | jq '.count, .history[0:2]'
echo ""

echo "Test 8: Today's Analytics"
echo "------------------------"
curl -s "http://localhost:8101/api/v1/analytics?date=$(date +%Y-%m-%d)" | jq '.analytics.total_queries, .analytics.intent_distribution'
echo ""

echo "Test 9: Entity Extraction (Testing en_core_web_lg)"
echo "---------------------------------------------------"
curl -s -X POST http://localhost:8101/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple released the iPhone 15 in California"}' \
  | jq '.analysis.entities'
echo ""

echo "Test 10: Caching (Run same query twice to test cache)"
echo "-----------------------------------------------------"
echo "First request (cold):"
time curl -s -X POST http://localhost:8101/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Test cache performance"}' \
  | jq '.processing_time_ms'
echo ""
echo "Second request (should be cached):"
time curl -s -X POST http://localhost:8101/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Test cache performance"}' \
  | jq '.processing_time_ms'
echo ""

echo "========================================="
echo "All tests completed!"
echo "========================================="
