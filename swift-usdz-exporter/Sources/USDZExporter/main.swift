import Foundation
import SceneKit

let sceneRotationRadians = -CGFloat.pi / 2.0

struct MeshManifestEntry: Decodable {
    let name: String
    let obj_file: String
    let color: String?
    let source_file: String
    let expression_id: String?
    let family: String
}

struct MeshManifest: Decodable {
    let schema_version: Int
    let mesh_count: Int
    let failed_mesh_count: Int
    let meshes: [MeshManifestEntry]
}

enum ExporterError: Error, CustomStringConvertible {
    case usage(String)
    case invalidManifestPath(String)
    case decodeFailed(String)
    case exportFailed(String)

    var description: String {
        switch self {
        case .usage(let msg), .invalidManifestPath(let msg), .decodeFailed(let msg), .exportFailed(let msg):
            return msg
        }
    }
}

struct Config {
    let manifestPath: URL
    let outputPath: URL
}

func parseArgs(_ args: [String]) throws -> Config {
    var manifest: String?
    var output: String?

    var i = 1
    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--manifest":
            i += 1
            guard i < args.count else { throw ExporterError.usage("Missing value for --manifest") }
            manifest = args[i]
        case "--output":
            i += 1
            guard i < args.count else { throw ExporterError.usage("Missing value for --output") }
            output = args[i]
        case "-h", "--help":
            throw ExporterError.usage(usage())
        default:
            throw ExporterError.usage("Unknown argument: \(arg)\n\n\(usage())")
        }
        i += 1
    }

    guard let manifest, let output else {
        throw ExporterError.usage(usage())
    }

    return Config(
        manifestPath: URL(fileURLWithPath: manifest),
        outputPath: URL(fileURLWithPath: output)
    )
}

func usage() -> String {
    return """
    usdz-exporter --manifest <path/to/manifest.json> --output <path/to/output.usdz>

    Example:
      swift run usdz-exporter --manifest ../artifacts/bridge/JSONreference/manifest.json --output ../artifacts/usdz/JSONreference.usdz
    """
}

func loadManifest(from url: URL) throws -> MeshManifest {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw ExporterError.invalidManifestPath("Manifest not found: \(url.path)")
    }

    do {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MeshManifest.self, from: data)
    } catch {
        throw ExporterError.decodeFailed("Failed to parse manifest: \(error.localizedDescription)")
    }
}

func colorFromHex(_ hex: String?) -> NSColor {
    guard let hex else { return NSColor.lightGray }
    let cleaned = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
    guard cleaned.count == 6, let intVal = Int(cleaned, radix: 16) else { return NSColor.lightGray }

    let r = CGFloat((intVal >> 16) & 0xFF) / 255.0
    let g = CGFloat((intVal >> 8) & 0xFF) / 255.0
    let b = CGFloat(intVal & 0xFF) / 255.0
    return NSColor(calibratedRed: r, green: g, blue: b, alpha: 1.0)
}

func applyMaterial(_ node: SCNNode, color: NSColor) {
    if let geometry = node.geometry {
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.lightingModel = .physicallyBased
        geometry.materials = [material]
    }
    for child in node.childNodes {
        applyMaterial(child, color: color)
    }
}

func loadNode(from objURL: URL, color: NSColor) -> SCNNode? {
    guard FileManager.default.fileExists(atPath: objURL.path) else {
        return nil
    }
    guard let sceneSource = SCNSceneSource(url: objURL, options: nil),
          let scene = sceneSource.scene(options: nil) else {
        return nil
    }
    let container = SCNNode()
    scene.rootNode.childNodes.forEach { child in
        applyMaterial(child, color: color)
        container.addChildNode(child)
    }
    return container
}

func exportUSDZ(config: Config) throws {
    let manifest = try loadManifest(from: config.manifestPath)
    let scene = SCNScene()
    let manifestDir = config.manifestPath.deletingLastPathComponent()
    let worldNode = SCNNode()
    scene.rootNode.addChildNode(worldNode)

    var loadedCount = 0
    for mesh in manifest.meshes {
        let objURL = manifestDir.appendingPathComponent(mesh.obj_file)
        let color = colorFromHex(mesh.color)
        if let node = loadNode(from: objURL, color: color) {
            node.name = mesh.name
            worldNode.addChildNode(node)
            loadedCount += 1
        } else {
            fputs("Warning: failed to load OBJ at \(objURL.path)\n", stderr)
        }
    }

    if loadedCount == 0 {
        throw ExporterError.exportFailed("No meshes were loaded from manifest. Cannot export USDZ.")
    }

    // Global correction: rotate model 90 degrees CCW around X and lift above x-z plane.
    worldNode.eulerAngles.x = sceneRotationRadians
    var minBounds = SCNVector3Zero
    var maxBounds = SCNVector3Zero
    let hasBounds = worldNode.__getBoundingBoxMin(&minBounds, max: &maxBounds)
    if hasBounds && minBounds.y < 0 {
        worldNode.position.y = -minBounds.y
    }

    let outputDir = config.outputPath.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let ok = scene.write(
        to: config.outputPath,
        options: nil,
        delegate: nil,
        progressHandler: nil
    )

    if !ok {
        throw ExporterError.exportFailed("SceneKit failed to write USDZ at: \(config.outputPath.path)")
    }

    print("USDZ export complete")
    print("- output: \(config.outputPath.path)")
    print("- meshes in manifest: \(manifest.mesh_count)")
    print("- meshes loaded: \(loadedCount)")
    print("- failed meshes from Python stage: \(manifest.failed_mesh_count)")
}

@main
struct USDZExporterMain {
    static func main() {
        do {
            let config = try parseArgs(CommandLine.arguments)
            try exportUSDZ(config: config)
        } catch let err as ExporterError {
            fputs("Error: \(err.description)\n", stderr)
            exit(1)
        } catch {
            fputs("Unexpected error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
