#!/bin/bash

# Check if a version is passed
if [ $# -eq 0 ]; then
    echo "No version specified. Usage: ./release_ai.sh <version>"
    exit 1
fi

VERSION=$1

# Step 0: Ensure everything is committed and pushed
# echo "Checking for uncommitted changes..."
# if [ -n "$(git status --porcelain)" ]; then
#     echo "Uncommitted changes detected. Please commit and push all changes before making a release."
#     exit 1
# fi

echo "Checking if current branch is up to date..."
git fetch
HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse main@{upstream})

# if [ "$HEADHASH" != "$UPSTREAMHASH" ]; then
#     echo "Current branch is not up to date with origin. Please push all changes before making a release."
#     exit 1
# fi

echo "Good to go!"

# # Step 1: Update setup.py with the new version
# echo "Updating setup.py with version $VERSION..."
# sed -i '' "s/version='.*'/version='$VERSION'/" setup.py
sed -i '' "s/__version__ = '[^']*'/__version__ = '$VERSION'/" ai/version.py




# Step 2: Build the Package
echo "Building the ai package..."
python setup.py sdist bdist_wheel  # Build the package

# # Step 3: Push the build to GitHub and create a release
echo "Pushing to GitHub and creating a release..."

# Define variables
REPO="aelaguiz/manbot_ai"  # Change to your GitHub username and repository
RELEASE_NAME="ai-v$VERSION"
WHEEL_FILE="dist/ai-$VERSION-py3-none-any.whl"

# Make sure the GitHub CLI is installed and configured
# Create a new release and upload the wheel file
echo gh release create $RELEASE_NAME $WHEEL_FILE -t "$RELEASE_NAME" -n "Release of version $VERSION"
gh release create $RELEASE_NAME $WHEEL_FILE -t "$RELEASE_NAME" -n "Release of version $VERSION"

# Step 4: Output the necessary information for the backend to import
echo "To import the new ai version into backend, update requirements.txt with:"
echo "ai @ https://github.com/$REPO/releases/download/$RELEASE_NAME/ai-$VERSION-py3-none-any.whl"