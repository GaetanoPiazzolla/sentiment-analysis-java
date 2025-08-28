package gae.piaz.sentiment.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Arrays;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;

@ExtendWith(MockitoExtension.class)
class SentimentAnalyzerServiceTest {

    @Mock
    private Predictor<String, Classifications> predictor;

    private SentimentAnalyzerService sentimentAnalyzerService;

    @BeforeEach
    void setUp() {
        sentimentAnalyzerService = new SentimentAnalyzerService(predictor);
    }

    @Test
    void testAnalyzeSentimentSimpleReturnsScore() throws Exception {
        // Arrange
        Classifications.Classification positive = mock(Classifications.Classification.class);
        when(positive.getClassName()).thenReturn("positive");
        when(positive.getProbability()).thenReturn(0.7);

        Classifications.Classification neutral = mock(Classifications.Classification.class);
        when(neutral.getClassName()).thenReturn("neutral");
        when(neutral.getProbability()).thenReturn(0.2);

        Classifications.Classification negative = mock(Classifications.Classification.class);
        when(negative.getClassName()).thenReturn("negative");
        when(negative.getProbability()).thenReturn(0.1);

        Classifications result = mock(Classifications.class);
        when(result.items()).thenReturn(Arrays.asList(positive, neutral, negative));
        when(predictor.predict(anyString())).thenReturn(result);

        // Act
        Double sentimentScore = sentimentAnalyzerService.analyzeSentimentSimple("The company reported excellent quarterly earnings");

        // Assert
        assertNotNull(sentimentScore, "Sentiment score should not be null");
        assertTrue(sentimentScore > 0, "Should be positive sentiment");
    }
}
