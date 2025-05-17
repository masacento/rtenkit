
# rtenkit go bindings

## Example

See [example/main.go](example/main.go)

```
cd example
make modeldata
make example_static
./example_static
make dynamic_static
./example_dynamic
```

## Tasks

### darwin_dylib
```
install_name_tool -id librtenkit.dylib ../../dist/rtenkit-darwin-arm64/librtenkit.dylib
otool -L ../../dist/rtenkit-darwin-arm64/librtenkit.dylib
```