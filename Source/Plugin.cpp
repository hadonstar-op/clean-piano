// Clean Piano - a simple additive piano synth (JUCE, VST3)
// Sound: inharmonic partials, two slightly detuned strings per note,
// hammer-position comb, velocity-dependent brightness, damper release,
// sustain pedal (handled by juce::Synthesiser), light reverb.

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <cmath>

namespace
{
    constexpr int kMaxPartials = 28;
    constexpr int kNumVoices   = 24;
    constexpr double kTwoPi    = 6.283185307179586;

    struct PianoSound : public juce::SynthesiserSound
    {
        bool appliesToNote (int) override    { return true; }
        bool appliesToChannel (int) override { return true; }
    };

    class PianoVoice : public juce::SynthesiserVoice
    {
    public:
        explicit PianoVoice (juce::AudioProcessorValueTreeState& s) : apvts (s) {}

        bool canPlaySound (juce::SynthesiserSound* s) override
        {
            return dynamic_cast<PianoSound*> (s) != nullptr;
        }

        void startNote (int midi, float vel, juce::SynthesiserSound*, int) override
        {
            const double sr = getSampleRate();
            const float decayParam = *apvts.getRawParameterValue ("decay");
            const float bright     = *apvts.getRawParameterValue ("brightness");
            const float hardness   = *apvts.getRawParameterValue ("hardness");
            const float width      = *apvts.getRawParameterValue ("width");

            const double f0 = 440.0 * std::pow (2.0, (midi - 69) / 12.0);

            // Inharmonicity grows toward the treble/bass extremes of real strings
            double B = 0.00006 * std::exp ((midi - 40) * 0.045);
            B = juce::jmin (B, 0.02);

            // Spectral tilt: harder hits and higher "brightness" => more overtones
            const float tilt = 2.4f - 1.7f * bright * (0.35f + 0.65f * vel);

            // Fundamental T60: lower notes ring longer
            const float T1 = juce::jlimit (0.3f, 30.0f,
                                           decayParam * (float) std::pow (2.0, -(midi - 60) / 18.0));

            numPartials = 0;
            for (int n = 1; n <= kMaxPartials; ++n)
            {
                const double f = n * f0 * std::sqrt (1.0 + B * n * n);
                if (f > 0.45 * sr) break;

                const float comb = 0.25f + std::abs (std::sin ((float) juce::MathConstants<double>::pi * n / 8.0f));
                const float a    = 0.5f * comb / std::pow ((float) n, tilt);

                const float Tn = T1 / (1.0f + 0.55f * std::pow ((float) (n - 1), 0.9f));
                dec[numPartials] = (float) std::exp (-6.9078 / (Tn * sr));
                amp[numPartials] = a;

                const double spread = 0.0004;
                inc[numPartials][0] = kTwoPi * f * (1.0 - spread) / sr;
                inc[numPartials][1] = kTwoPi * f * (1.0 + spread) / sr;
                ph[numPartials][0] = ph[numPartials][1] = 0.0;
                ++numPartials;
            }

            // Stereo placement: notes spread slightly across keyboard, strings spread by width
            const float notePan = juce::jlimit (-1.0f, 1.0f, (midi - 60) / 48.0f) * 0.35f;
            for (int s = 0; s < 2; ++s)
            {
                const float pan = juce::jlimit (-1.0f, 1.0f, notePan + (s == 0 ? -1.0f : 1.0f) * width * 0.5f);
                const float ang = (pan + 1.0f) * 0.25f * juce::MathConstants<float>::pi;
                panL[s] = std::cos (ang);
                panR[s] = std::sin (ang);
            }

            masterGain  = std::pow (vel, 1.3f) * 0.22f;
            attackGain  = 0.0f;
            attackInc   = (float) (1.0 / (0.002 * sr));
            relMul      = (float) std::exp (-6.9078 / (0.25 * sr));
            releasing   = false;

            noiseEnv    = vel * hardness * 0.35f;
            noiseDec    = (float) std::exp (-1.0 / (0.012 * sr));
            noiseLP     = 0.0f;
            noiseCoef   = juce::jlimit (0.05f, 0.9f, 0.1f + 0.7f * vel);
        }

        void stopNote (float, bool allowTailOff) override
        {
            if (allowTailOff)
                releasing = true;
            else
            {
                clearCurrentNote();
                numPartials = 0;
            }
        }

        void pitchWheelMoved (int) override {}
        void controllerMoved (int, int) override {}

        void renderNextBlock (juce::AudioBuffer<float>& out, int start, int num) override
        {
            if (numPartials == 0)
                return;

            auto* l = out.getWritePointer (0, start);
            auto* r = out.getNumChannels() > 1 ? out.getWritePointer (1, start) : nullptr;

            for (int i = 0; i < num; ++i)
            {
                float sumL = 0.0f, sumR = 0.0f;

                for (int p = 0; p < numPartials; ++p)
                {
                    amp[p] *= dec[p];
                    if (releasing) amp[p] *= relMul;

                    for (int s = 0; s < 2; ++s)
                    {
                        ph[p][s] += inc[p][s];
                        if (ph[p][s] > kTwoPi) ph[p][s] -= kTwoPi;
                        const float v = amp[p] * (float) std::sin (ph[p][s]);
                        sumL += v * panL[s];
                        sumR += v * panR[s];
                    }
                }

                // Hammer thump: short low-passed noise burst
                if (noiseEnv > 1.0e-5f)
                {
                    const float n = random.nextFloat() * 2.0f - 1.0f;
                    noiseLP += noiseCoef * (n - noiseLP);
                    const float h = noiseLP * noiseEnv;
                    sumL += h;
                    sumR += h;
                    noiseEnv *= noiseDec;
                }

                attackGain = juce::jmin (1.0f, attackGain + attackInc);
                const float g = masterGain * attackGain;

                l[i] += sumL * g;
                if (r != nullptr) r[i] += sumR * g;
            }

            if (amp[0] < 1.0e-5f)
            {
                clearCurrentNote();
                numPartials = 0;
            }
        }

    private:
        juce::AudioProcessorValueTreeState& apvts;
        juce::Random random;

        int numPartials = 0;
        float amp[kMaxPartials] {};
        float dec[kMaxPartials] {};
        double ph[kMaxPartials][2] {};
        double inc[kMaxPartials][2] {};
        float panL[2] { 0.7f, 0.7f }, panR[2] { 0.7f, 0.7f };

        float masterGain = 0.0f, attackGain = 0.0f, attackInc = 0.0f, relMul = 1.0f;
        bool releasing = false;

        float noiseEnv = 0.0f, noiseDec = 0.0f, noiseLP = 0.0f, noiseCoef = 0.3f;
    };
}

//==============================================================================
class CleanPianoProcessor : public juce::AudioProcessor
{
public:
    CleanPianoProcessor()
        : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
          apvts (*this, nullptr, "PARAMS", createLayout())
    {
        for (int i = 0; i < kNumVoices; ++i)
            synth.addVoice (new PianoVoice (apvts));
        synth.addSound (new PianoSound());
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using P = juce::AudioParameterFloat;
        using R = juce::NormalisableRange<float>;
        return {
            std::make_unique<P> (juce::ParameterID { "decay", 1 },      "Decay (s)",  R (0.5f, 12.0f, 0.01f, 0.5f), 4.0f),
            std::make_unique<P> (juce::ParameterID { "brightness", 1 }, "Brightness", R (0.0f, 1.0f, 0.001f), 0.5f),
            std::make_unique<P> (juce::ParameterID { "hardness", 1 },   "Hammer",     R (0.0f, 1.0f, 0.001f), 0.3f),
            std::make_unique<P> (juce::ParameterID { "width", 1 },      "Width",      R (0.0f, 1.0f, 0.001f), 0.5f),
            std::make_unique<P> (juce::ParameterID { "reverb", 1 },     "Reverb",     R (0.0f, 1.0f, 0.001f), 0.2f),
            std::make_unique<P> (juce::ParameterID { "volume", 1 },     "Volume",     R (0.0f, 1.0f, 0.001f), 0.7f)
        };
    }

    void prepareToPlay (double sampleRate, int) override
    {
        synth.setCurrentPlaybackSampleRate (sampleRate);
        reverb.setSampleRate (sampleRate);
        reverb.reset();
    }
    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override
    {
        juce::ScopedNoDenormals noDenormals;
        buffer.clear();
        synth.renderNextBlock (buffer, midi, 0, buffer.getNumSamples());

        const float rv = *apvts.getRawParameterValue ("reverb");
        juce::Reverb::Parameters rp;
        rp.roomSize   = 0.55f;
        rp.damping    = 0.55f;
        rp.wetLevel   = rv * 0.35f;
        rp.dryLevel   = 1.0f - rv * 0.25f;
        rp.width      = 1.0f;
        rp.freezeMode = 0.0f;
        reverb.setParameters (rp);

        if (buffer.getNumChannels() >= 2)
            reverb.processStereo (buffer.getWritePointer (0), buffer.getWritePointer (1), buffer.getNumSamples());

        const float vol = *apvts.getRawParameterValue ("volume");
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            auto* d = buffer.getWritePointer (ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                d[i] = std::tanh (d[i] * vol * 2.0f); // soft clip safety
        }
    }

    juce::AudioProcessorEditor* createEditor() override { return new juce::GenericAudioProcessorEditor (*this); }
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Clean Piano"; }
    bool acceptsMidi() const override  { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 6.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& dest) override
    {
        if (auto xml = apvts.copyState().createXml())
            copyXmlToBinary (*xml, dest);
    }
    void setStateInformation (const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary (data, size))
            if (xml->hasTagName (apvts.state.getType()))
                apvts.replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    juce::AudioProcessorValueTreeState apvts;
    juce::Synthesiser synth;
    juce::Reverb reverb;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CleanPianoProcessor)
};

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new CleanPianoProcessor();
}
