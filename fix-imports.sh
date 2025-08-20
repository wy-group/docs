#!/bin/bash

# Fix imports in all MDX files
find /Users/tian/Work/llm-doc/content/docs -name "*.mdx" -type f | while read -r file; do
    echo "Fixing imports in: $file"
    
    # Replace the incorrect import with correct ones
    sed -i '' "s/import { Callout, Card, Cards } from 'fumadocs-ui\/components\/callout';/import { Callout } from 'fumadocs-ui\/components\/callout';\nimport { Card, Cards } from 'fumadocs-ui\/components\/card';/g" "$file"
done

echo "✅ All imports fixed!"