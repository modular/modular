# Nightly: v0.26.3

This version is still a work in progress.

## ✨ Highlights

## Language enhancements

## Language changes

## Library changes

- The `DimList` type has moved to representing its dimensions as parameters to
  the type instead of values inside the type, directly reflecting that the
  dimensions are known at compile time.  Please change `DimList(x, y)`
  into `DimList[x, y]()`.

## Tooling changes

## ❌ Removed

## 🛠️ Fixed
