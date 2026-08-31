using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace TinyTransformer.Api.Tests;

public class EncodeEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public EncodeEndpointTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task Health_ReturnsHealthy()
    {
        var response = await _client.GetAsync("/api/health");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("status").GetString().Should().Be("healthy");
    }

    [Fact]
    public async Task Root_ServesTheFrontend()
    {
        var response = await _client.GetAsync("/");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        response.Content.Headers.ContentType?.MediaType.Should().Be("text/html");
    }

    [Fact]
    public async Task Encode_WithValidText_ReturnsMatchingShapes()
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", dModel = 8, dK = 4, ffHidden = 16, seed = 1 });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        int sequenceLength = body.GetProperty("config").GetProperty("sequenceLength").GetInt32();
        int dModel = body.GetProperty("config").GetProperty("dModel").GetInt32();

        sequenceLength.Should().Be(2); // "hi" -> 2 characters
        dModel.Should().Be(8);
        body.GetProperty("tokens").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("embeddings").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("embeddings")[0].GetArrayLength().Should().Be(dModel);
        body.GetProperty("attentionWeights").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("attentionWeights")[0].GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("encoderOutput").GetArrayLength().Should().Be(sequenceLength);
    }

    [Fact]
    public async Task Encode_WithTheSameSeed_IsDeterministic()
    {
        var request = new { text = "same seed", dModel = 8, dK = 4, ffHidden = 16, seed = 123 };

        var first = await (await _client.PostAsJsonAsync("/api/encode", request)).Content.ReadAsStringAsync();
        var second = await (await _client.PostAsJsonAsync("/api/encode", request)).Content.ReadAsStringAsync();

        first.Should().Be(second);
    }

    [Fact]
    public async Task Encode_WithEmptyText_ReturnsValidationProblem()
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "" });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("errors").TryGetProperty("text", out _).Should().BeTrue();
    }

    [Fact]
    public async Task Encode_WithTextTooLong_ReturnsValidationProblem()
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = new string('a', 65) });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("errors").TryGetProperty("text", out _).Should().BeTrue();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1000)]
    public async Task Encode_WithDModelOutOfRange_ReturnsValidationProblem(int dModel)
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", dModel });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("errors").TryGetProperty("dModel", out _).Should().BeTrue();
    }

    [Fact]
    public async Task Encode_WithMultipleHeads_ReturnsMatchingShapes()
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", dModel = 8, dK = 4, ffHidden = 16, numHeads = 4, seed = 1 });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        int sequenceLength = body.GetProperty("config").GetProperty("sequenceLength").GetInt32();
        int dModel = body.GetProperty("config").GetProperty("dModel").GetInt32();

        body.GetProperty("config").GetProperty("numHeads").GetInt32().Should().Be(4);
        // The top-level attention-weights shape doesn't change with numHeads -
        // multiple heads are averaged into the same [T x T] matrix (see
        // TransformerEncoderBlock.ForwardWithAttention).
        body.GetProperty("attentionWeights").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("attentionWeights")[0].GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("encoderOutput").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("encoderOutput")[0].GetArrayLength().Should().Be(dModel);

        // attentionWeightsPerLayer is [layer][head][T][T] - one layer here,
        // 4 unaveraged heads.
        var perLayer = body.GetProperty("attentionWeightsPerLayer");
        perLayer.GetArrayLength().Should().Be(1);
        perLayer[0].GetArrayLength().Should().Be(4);
        perLayer[0][0].GetArrayLength().Should().Be(sequenceLength);
        perLayer[0][0][0].GetArrayLength().Should().Be(sequenceLength);
    }

    [Fact]
    public async Task Encode_WithMultipleLayers_ReturnsMatchingShapes()
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", dModel = 8, dK = 4, ffHidden = 16, numLayers = 3, seed = 1 });

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();

        int sequenceLength = body.GetProperty("config").GetProperty("sequenceLength").GetInt32();
        int dModel = body.GetProperty("config").GetProperty("dModel").GetInt32();

        body.GetProperty("config").GetProperty("numLayers").GetInt32().Should().Be(3);
        body.GetProperty("attentionWeights").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("attentionWeights")[0].GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("encoderOutput").GetArrayLength().Should().Be(sequenceLength);
        body.GetProperty("encoderOutput")[0].GetArrayLength().Should().Be(dModel);

        // attentionWeightsPerLayer is [layer][head][T][T] - 3 layers here,
        // one (unaveraged) head each.
        var perLayer = body.GetProperty("attentionWeightsPerLayer");
        perLayer.GetArrayLength().Should().Be(3);
        perLayer[0].GetArrayLength().Should().Be(1);
        perLayer[0][0].GetArrayLength().Should().Be(sequenceLength);
        perLayer[0][0][0].GetArrayLength().Should().Be(sequenceLength);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(100)]
    public async Task Encode_WithNumHeadsOutOfRange_ReturnsValidationProblem(int numHeads)
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", numHeads });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("errors").TryGetProperty("numHeads", out _).Should().BeTrue();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(100)]
    public async Task Encode_WithNumLayersOutOfRange_ReturnsValidationProblem(int numLayers)
    {
        var response = await _client.PostAsJsonAsync("/api/encode", new { text = "hi", numLayers });

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("errors").TryGetProperty("numLayers", out _).Should().BeTrue();
    }
}
