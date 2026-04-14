// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "USDZExporter",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "usdz-exporter", targets: ["USDZExporter"]),
    ],
    targets: [
        .executableTarget(
            name: "USDZExporter",
            path: "Sources/USDZExporter"
        ),
        .testTarget(
            name: "USDZExporterTests",
            dependencies: ["USDZExporter"],
            path: "Tests/USDZExporterTests"
        )
    ]
)
