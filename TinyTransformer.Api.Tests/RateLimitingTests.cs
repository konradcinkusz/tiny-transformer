using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace TinyTransformer.Api.Tests;

// Its own factory instance (not IClassFixture-shared with EncodeEndpointTests)
// so this test's burst of requests cannot push the fixed-window counter past
// 30 for unrelated tests sharing the same in-memory client/IP partition.
public class RateLimitingTests : IDisposable
{
    private readonly WebApplicationFactory<Program> _factory = new();

    [Fact]
    public async Task Encode_RejectsRequestsAboveTheFixedWindowLimit()
    {
        var client = _factory.CreateClient();
        var request = new { text = "rate limit probe" };

        HttpResponseMessage? rejected = null;
        for (int i = 0; i < 40 && rejected is null; i++)
        {
            var response = await client.PostAsJsonAsync("/api/encode", request);
            if (response.StatusCode == HttpStatusCode.TooManyRequests)
                rejected = response;
        }

        rejected.Should().NotBeNull("the fixed-window limiter allows only 30 requests per client per minute");
        var body = await rejected!.Content.ReadFromJsonAsync<JsonElement>();
        body.GetProperty("error").GetString().Should().NotBeNullOrEmpty();
    }

    public void Dispose() => _factory.Dispose();
}
